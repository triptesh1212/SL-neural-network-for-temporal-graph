from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from utils.utils import NeighborFinderAdapter, TimeEncoder


class MergeLayer(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads

        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        assert self.query_dim % num_heads == 0
        self.head_dim = self.query_dim // num_heads

        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.scaling_factor = self.head_dim ** -0.5
        self.layer_norm = nn.LayerNorm(self.query_dim)
        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_time_features, neighbor_node_features,
                neighbor_node_time_features, neighbor_node_edge_features, neighbor_masks):
        node_features = torch.unsqueeze(node_features, dim=1)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)

        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = torch.einsum('bhld,bhnd->bhln', query, key) * self.scaling_factor
        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1) == 0
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)
        attention = attention.masked_fill(attention_mask, -1e10)
        attention_scores = self.dropout(torch.softmax(attention, dim=-1))
        attention_output = torch.einsum('bhln,bhnd->bhld', attention_scores, value)
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)
        output = self.dropout(self.residual_fc(attention_output))
        output = self.layer_norm(output + residual).squeeze(dim=1)
        return output, attention_scores.squeeze(dim=2)


class MemoryModel(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, model_name: str = 'TGN', num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, src_node_mean_time_shift: float = 0.0,
                 src_node_std_time_shift: float = 1.0, dst_node_mean_time_shift_dst: float = 0.0,
                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu'):
        super().__init__()

        self.register_buffer('node_raw_features', torch.from_numpy(node_raw_features.astype(np.float32)))
        self.register_buffer('edge_raw_features', torch.from_numpy(edge_raw_features.astype(np.float32)))

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift

        self.model_name = model_name
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.message_aggregator = MessageAggregator()
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)

        if self.model_name == 'TGN':
            self.memory_updater = GRUMemoryUpdater(
                memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        elif self.model_name in ['DyRep', 'JODIE']:
            self.memory_updater = RNNMemoryUpdater(
                memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        if self.model_name == 'JODIE':
            self.embedding_module = TimeProjectionEmbedding(memory_dim=self.memory_dim, dropout=self.dropout)
        elif self.model_name in ['TGN', 'DyRep']:
            self.embedding_module = GraphAttentionEmbedding(
                neighbor_sampler=neighbor_sampler,
                time_encoder=self.time_encoder,
                node_feat_dim=self.node_feat_dim,
                edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=self.time_feat_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout)
            # share feature buffers with the parent so .to(device) stays consistent
            self.embedding_module.node_raw_features = self.node_raw_features
            self.embedding_module.edge_raw_features = self.edge_raw_features
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, edge_ids: np.ndarray = None,
                                                 edges_are_positive: bool = True, num_neighbors: int = 20):
        node_ids = np.concatenate([src_node_ids, dst_node_ids])
        updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(
            node_ids=np.array(range(self.num_nodes)),
            node_raw_messages=self.memory_bank.node_raw_messages)

        if self.model_name == 'JODIE':
            src_node_time_intervals = (
                torch.from_numpy(node_interact_times).float().to(self.device)
                - updated_node_last_updated_times[torch.from_numpy(src_node_ids)]
            )
            src_node_time_intervals = (
                (src_node_time_intervals - self.src_node_mean_time_shift) / self.src_node_std_time_shift
            )
            dst_node_time_intervals = (
                torch.from_numpy(node_interact_times).float().to(self.device)
                - updated_node_last_updated_times[torch.from_numpy(dst_node_ids)]
            )
            dst_node_time_intervals = (
                (dst_node_time_intervals - self.dst_node_mean_time_shift_dst) / self.dst_node_std_time_shift
            )
            node_time_intervals = torch.cat([src_node_time_intervals, dst_node_time_intervals], dim=0)
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
                node_memories=updated_node_memories, node_ids=node_ids, node_time_intervals=node_time_intervals)
        elif self.model_name in ['TGN', 'DyRep']:
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
                node_memories=updated_node_memories,
                node_ids=node_ids,
                node_interact_times=np.concatenate([node_interact_times, node_interact_times]),
                current_layer_num=self.num_layers,
                num_neighbors=num_neighbors)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        src_node_embeddings = node_embeddings[:len(src_node_ids)]
        dst_node_embeddings = node_embeddings[len(src_node_ids): len(src_node_ids) + len(dst_node_ids)]

        if edges_are_positive:
            assert edge_ids is not None
            self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(
                src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                dst_node_embeddings=dst_node_embeddings,
                node_interact_times=node_interact_times, edge_ids=edge_ids)
            unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(
                src_node_ids=dst_node_ids, dst_node_ids=src_node_ids,
                dst_node_embeddings=src_node_embeddings,
                node_interact_times=node_interact_times, edge_ids=edge_ids)

            self.memory_bank.store_node_raw_messages(
                node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
            self.memory_bank.store_node_raw_messages(
                node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        # DyRep uses updated memories as embeddings (attention is used only for messages)
        if self.model_name == 'DyRep':
            src_node_embeddings = updated_node_memories[torch.from_numpy(src_node_ids)]
            dst_node_embeddings = updated_node_memories[torch.from_numpy(dst_node_ids)]

        return src_node_embeddings, dst_node_embeddings

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        unique_node_ids, unique_node_messages, unique_node_timestamps = \
            self.message_aggregator.aggregate_messages(node_ids=node_ids, node_raw_messages=node_raw_messages)
        return self.memory_updater.get_updated_memories(
            unique_node_ids=unique_node_ids,
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps)

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        unique_node_ids, unique_node_messages, unique_node_timestamps = \
            self.message_aggregator.aggregate_messages(node_ids=node_ids, node_raw_messages=node_raw_messages)
        self.memory_updater.update_memories(
            unique_node_ids=unique_node_ids,
            unique_node_messages=unique_node_messages,
            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                      dst_node_embeddings: torch.Tensor, node_interact_times: np.ndarray,
                                      edge_ids: np.ndarray):
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        if self.model_name == 'DyRep':
            dst_node_memories = dst_node_embeddings
        else:
            dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        src_node_delta_times = (
            torch.from_numpy(node_interact_times).float().to(self.device)
            - self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        )
        src_node_delta_time_features = self.time_encoder(
            src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]
        new_src_node_raw_messages = torch.cat(
            [src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        new_node_raw_messages = defaultdict(list)
        unique_node_ids = np.unique(src_node_ids)
        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append(
                (new_src_node_raw_messages[i], node_interact_times[i]))
        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler):
        assert self.model_name in ['TGN', 'DyRep'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.embedding_module.neighbor_sampler = neighbor_sampler
        # keep embedding module feature views synced with registered buffers
        self.embedding_module.node_raw_features = self.node_raw_features
        self.embedding_module.edge_raw_features = self.edge_raw_features
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            if self.embedding_module.neighbor_sampler.seed is not None:
                self.embedding_module.neighbor_sampler.reset_random_state()


class MessageAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []
        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        to_update_node_ids = np.array(to_update_node_ids)
        unique_node_messages = (
            torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        )
        unique_node_timestamps = np.array(unique_node_timestamps)
        return to_update_node_ids, unique_node_messages, unique_node_timestamps


class MemoryBank(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        self.node_raw_messages = defaultdict(list)
        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [
                (msg[0].clone(), np.array(msg[1], copy=True)) for msg in node_raw_messages
            ]
        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        self.node_memories.data = backup_memory_bank[0].clone()
        self.node_last_updated_times.data = backup_memory_bank[1].clone()
        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [
                (msg[0].clone(), np.array(msg[1], copy=True)) for msg in node_raw_messages
            ]

    def detach_memory_bank(self):
        self.node_memories.detach_()
        for node_id, node_raw_messages in self.node_raw_messages.items():
            self.node_raw_messages[node_id] = [(msg[0].detach(), msg[1]) for msg in node_raw_messages]

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):
    def __init__(self, memory_bank: MemoryBank):
        super().__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        if len(unique_node_ids) <= 0:
            return
        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), \
            "Trying to update memory to time in the past!"
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = (
            torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)
        )

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()
        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), \
            "Trying to update memory to time in the past!"
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(
            unique_node_messages, updated_node_memories[torch.from_numpy(unique_node_ids)])
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = (
            torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)
        )
        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        super().__init__(memory_bank)
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):
    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        super().__init__(memory_bank)
        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)


class TimeProjectionEmbedding(nn.Module):
    def __init__(self, memory_dim: int, dropout: float):
        super().__init__()
        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray,
                                         node_time_intervals: torch.Tensor):
        return self.dropout(
            node_memories[torch.from_numpy(node_ids)]
            * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1)))
        )


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, neighbor_sampler, time_encoder: TimeEncoder, node_feat_dim: int, edge_feat_dim: int,
                 time_feat_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_raw_features = None
        self.edge_raw_features = None
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_conv_layers = nn.ModuleList([
            MultiHeadAttention(
                node_feat_dim=self.node_feat_dim, edge_feat_dim=self.edge_feat_dim,
                time_feat_dim=self.time_feat_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(num_layers)
        ])
        self.merge_layers = nn.ModuleList([
            MergeLayer(
                input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim)
            for _ in range(num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray,
                                         node_interact_times: np.ndarray, current_layer_num: int,
                                         num_neighbors: int = 20):
        assert current_layer_num >= 0
        device = self.node_raw_features.device
        node_time_features = self.time_encoder(
            timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_features

        node_conv_features = self.compute_node_temporal_embeddings(
            node_memories=node_memories, node_ids=node_ids, node_interact_times=node_interact_times,
            current_layer_num=current_layer_num - 1, num_neighbors=num_neighbors)

        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        neighbor_node_conv_features = self.compute_node_temporal_embeddings(
            node_memories=node_memories, node_ids=neighbor_node_ids.flatten(),
            node_interact_times=neighbor_times.flatten(),
            current_layer_num=current_layer_num - 1, num_neighbors=num_neighbors)
        neighbor_node_conv_features = neighbor_node_conv_features.reshape(
            node_ids.shape[0], num_neighbors, self.node_feat_dim)

        neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times
        neighbor_time_features = self.time_encoder(
            timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]

        output, _ = self.temporal_conv_layers[current_layer_num - 1](
            node_features=node_conv_features,
            node_time_features=node_time_features,
            neighbor_node_features=neighbor_node_conv_features,
            neighbor_node_time_features=neighbor_time_features,
            neighbor_node_edge_features=neighbor_edge_features,
            neighbor_masks=neighbor_node_ids)
        return self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)


def compute_src_dst_node_time_shifts(src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                     node_interact_times: np.ndarray):
    src_node_last_timestamps = dict()
    dst_node_last_timestamps = dict()
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []
    for k in range(len(src_node_ids)):
        src_node_id = src_node_ids[k]
        dst_node_id = dst_node_ids[k]
        node_interact_time = node_interact_times[k]
        if src_node_id not in src_node_last_timestamps:
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps:
            dst_node_last_timestamps[dst_node_id] = 0
        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])
        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time
    return (
        np.mean(src_node_all_time_shifts),
        np.std(src_node_all_time_shifts),
        np.mean(dst_node_all_time_shifts),
        np.std(dst_node_all_time_shifts),
    )


class MemoryLP(nn.Module):

    def __init__(
        self,
        ngh_finder,
        n_feat: np.ndarray,
        e_feat: np.ndarray,
        model_name: str,
        time_feat_dim: int = 100,
        num_layers: int = 1,
        num_heads: int = 2,
        drop_out: float = 0.1,
        src_node_mean_time_shift: float = 0.0,
        src_node_std_time_shift: float = 1.0,
        dst_node_mean_time_shift_dst: float = 0.0,
        dst_node_std_time_shift: float = 1.0,
        device: str = 'cpu',
    ):
        super().__init__()
        self._ngh_finder = ngh_finder
        self.model_name = model_name
        self.device = device
        neighbor_sampler = NeighborFinderAdapter(ngh_finder) if model_name in ['DyRep', 'TGN'] else None
        self.backbone = MemoryModel(
            node_raw_features=n_feat,
            edge_raw_features=e_feat,
            neighbor_sampler=neighbor_sampler,
            time_feat_dim=time_feat_dim,
            model_name=model_name,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=drop_out,
            src_node_mean_time_shift=src_node_mean_time_shift,
            src_node_std_time_shift=src_node_std_time_shift,
            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
            dst_node_std_time_shift=dst_node_std_time_shift,
            device=device,
        )
        node_feat_dim = n_feat.shape[1]
        self.link_predictor = MergeLayer(
            input_dim1=node_feat_dim, input_dim2=node_feat_dim,
            hidden_dim=node_feat_dim, output_dim=1,
        )

    @property
    def ngh_finder(self):
        return self._ngh_finder

    @ngh_finder.setter
    def ngh_finder(self, finder):
        self._ngh_finder = finder
        if self.model_name in ['DyRep', 'TGN']:
            self.backbone.set_neighbor_sampler(NeighborFinderAdapter(finder))

    @property
    def memory_bank(self):
        return self.backbone.memory_bank

    def _sync_device(self):
        device = self.backbone.node_raw_features.device
        self.backbone.device = str(device)
        if self.model_name in ['DyRep', 'TGN']:
            self.backbone.embedding_module.node_raw_features = self.backbone.node_raw_features
            self.backbone.embedding_module.edge_raw_features = self.backbone.edge_raw_features

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, edge_idx_l=None, num_neighbors=20):
        self._sync_device()
        src = np.asarray(src_idx_l)
        dst = np.asarray(target_idx_l)
        neg_dst = np.asarray(background_idx_l)
        ts = np.asarray(cut_time_l)
        edge_ids = None if edge_idx_l is None else np.asarray(edge_idx_l)

        neg_src_emb, neg_dst_emb = self.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=neg_dst, node_interact_times=ts,
            edge_ids=None, edges_are_positive=False, num_neighbors=num_neighbors)
        src_emb, dst_emb = self.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=src, dst_node_ids=dst, node_interact_times=ts,
            edge_ids=edge_ids, edges_are_positive=True, num_neighbors=num_neighbors)

        pos_score = self.link_predictor(src_emb, dst_emb).squeeze(dim=-1).sigmoid()
        neg_score = self.link_predictor(neg_src_emb, neg_dst_emb).squeeze(dim=-1).sigmoid()
        return pos_score, neg_score
