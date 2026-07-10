import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        timestamps = timestamps.unsqueeze(dim=2)
        output = torch.cos(self.w(timestamps))

        return output


class MergeLayerTD(nn.Module):

    def __init__(self, input_dim1: int, input_dim2: int, input_dim3: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2 + input_dim3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor, input_3: torch.Tensor):
        x = torch.cat([input_1, input_2, input_3], dim=1)
        h = self.fc2(self.act(self.fc1(x)))
        return h


class NeighborFinderAdapter:
    def __init__(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.sample_neighbor_strategy = 'recent'
        self.seed = None

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        for node_id, interact_time in zip(node_ids, node_interact_times):
            ngh_idx, ngh_eidx, ngh_ts = self.ngh_finder.find_before(int(node_id), float(interact_time))
            nodes_neighbor_ids_list.append(np.asarray(ngh_idx))
            nodes_edge_ids_list.append(np.asarray(ngh_eidx))
            nodes_neighbor_times_list.append(np.asarray(ngh_ts))
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)


class DyGMamba(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1,gamma: float = 0.5, max_input_sequence_length: int = 512, max_interaction_times: int = 10, device: str = 'cpu'):
    
        super(DyGMamba, self).__init__()

        self.register_buffer('node_raw_features', torch.from_numpy(node_raw_features.astype(np.float32)))
        self.register_buffer('edge_raw_features', torch.from_numpy(edge_raw_features.astype(np.float32)))

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gamma = gamma
        self.max_input_sequence_length = max_input_sequence_length
        self.max_interaction_times = max_interaction_times
        self.device = device


        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NIFEncoder(nif_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })


        self.num_channels = 4
        feature_expansion_size = 2

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, out_features=self.node_feat_dim, bias=True)
        self.output_layer_t_diff = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                             out_features=self.node_feat_dim, bias=True)

        self.mamba = nn.ModuleList([
            Mamba(d_model=self.num_channels * self.channel_embedding_dim // feature_expansion_size,  # Model dimension d_model
                  d_state=16,  # SSM state expansion factor
                  d_conv=4,  # Local convolution width
                  expand=1,  # Block expansion factor
                  )
            for _ in range(self.num_layers)
        ])

        self.mamba_t_diff = nn.ModuleList([
            Mamba(d_model=int(self.gamma*self.channel_embedding_dim),  # Model dimension d_model
                  d_state=16,  # SSM state expansion factor
                  d_conv=4,  # Local convolution width
                  expand=1,  # Block expansion factor
                  )
            for _ in range(self.num_layers)
        ])

        self.projection_layer_t_diff = nn.Linear(in_features=self.time_feat_dim, out_features=int(self.gamma*self.channel_embedding_dim), bias=True)
        self.projection_layer_t_diff_up = nn.Linear(in_features=int(self.gamma*self.channel_embedding_dim),
                                                    out_features=self.num_channels * self.channel_embedding_dim // feature_expansion_size, bias=True)


        self.weightagg = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, 1)

        self.reduce_layer = nn.Linear(self.num_channels * self.channel_embedding_dim, self.num_channels * self.channel_embedding_dim // feature_expansion_size)

        self.channel_norm = nn.LayerNorm(self.num_channels * self.channel_embedding_dim // feature_expansion_size)
        self.channel_feedforward = FeedForwardNet(input_dim=self.num_channels * self.channel_embedding_dim // feature_expansion_size,
                                                  dim_expansion_factor=4,
                                                  dropout=self.dropout)

        self.neighbor_selection_layer = nn.Linear(self.num_channels * self.channel_embedding_dim // feature_expansion_size, self.num_channels * self.channel_embedding_dim // feature_expansion_size)


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
       
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)


        padded_time_diff_emb = self.time_modeling(src_node_ids, dst_node_ids, node_interact_times,
                                                          src_nodes_neighbor_ids_list, src_nodes_neighbor_times_list, self.time_encoder)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)


        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, src_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)


        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)


        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)


        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        src_patches_data = [src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features,
                            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features]
        dst_patches_data = [dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features,
                            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features]
        src_patches_data = torch.stack(src_patches_data, dim=2)
        dst_patches_data = torch.stack(dst_patches_data, dim=2)
        src_patches_data = src_patches_data.reshape(batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim)
        dst_patches_data = dst_patches_data.reshape(batch_size, dst_num_patches,
                                                self.num_channels * self.channel_embedding_dim)

        # reduce to channel embsize
        src_patches_data = self.reduce_layer(src_patches_data)
        dst_patches_data = self.reduce_layer(dst_patches_data)


        for mamba in self.mamba:
            src_patches_data = mamba(src_patches_data) + src_patches_data
            dst_patches_data = mamba(dst_patches_data) + dst_patches_data
            src_patches_data = self.channel_norm(src_patches_data)
            dst_patches_data = self.channel_norm(dst_patches_data)
            src_patches_data = self.channel_feedforward(src_patches_data) + src_patches_data
            dst_patches_data = self.channel_feedforward(dst_patches_data) + dst_patches_data

        padded_time_diff_emb = self.projection_layer_t_diff(padded_time_diff_emb)
        for mamba_t in self.mamba_t_diff:
            padded_time_diff_emb = mamba_t(padded_time_diff_emb) + padded_time_diff_emb

        src_weight = self.weightagg(src_patches_data).transpose(1, 2)
        dst_weight = self.weightagg(dst_patches_data).transpose(1, 2)

        src_patches_data_ = src_weight.matmul(src_patches_data).squeeze(dim=1)
        dst_patches_data_ = dst_weight.matmul(dst_patches_data).squeeze(dim=1)

        time_diff_emb = torch.mean(padded_time_diff_emb, dim=1)
        time_diff_emb_ = self.projection_layer_t_diff_up(time_diff_emb)

        # Tensor, shape (batch_size, 1, channel_embedding_dim)
        src_selection_param = (self.neighbor_selection_layer(dst_patches_data_) * time_diff_emb_).unsqueeze(1)
        dst_selection_param = (self.neighbor_selection_layer(src_patches_data_) * time_diff_emb_).unsqueeze(1)


        src_patches_data = torch.sum(src_patches_data * torch.nn.functional.softmax(torch.sum(src_selection_param * src_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)
        dst_patches_data = torch.sum(dst_patches_data * torch.nn.functional.softmax(torch.sum(dst_selection_param * dst_patches_data, dim=2), dim=1).unsqueeze(2), dim=1)


        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        time_diff_emb = self.output_layer_t_diff(time_diff_emb)
        return src_node_embeddings, dst_node_embeddings, time_diff_emb
    
    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):

        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'

        max_seq_length = max_input_sequence_length
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]


        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]


            if len(nodes_neighbor_ids_list[idx]) > 0:
                # left padding
                padded_nodes_neighbor_ids[idx, -len(nodes_neighbor_ids_list[idx])-1:-1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -len(nodes_edge_ids_list[idx])-1:-1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -len(nodes_neighbor_times_list[idx])-1:-1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def find_previous_interaction(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray, dst_node_interact_times: np.ndarray,
                                  src_nodes_neighbor_ids_list: list, dst_nodes_neighbor_ids_list: list, src_nodes_edge_ids_list: list, dst_nodes_edge_ids_list: list,
                                  src_nodes_neighbor_times_list: list, dst_nodes_neighbor_times_list: list):
        src_latest_time_interaction, dst_latest_time_interaction = [], []
        gamma = 100
        shrink_ratio = 1e8
        shrink_coeff = torch.tensor(1/(gamma * shrink_ratio), device=self.node_raw_features.device)

        for idx in range(len(src_node_ids)):

            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)

            find_interact_index = np.nonzero(find_interact)

            if find_interact_index[0].shape[0] == 0: # previous interaction not found
                src_latest_time_interaction.append(0.0)
            else:

                src_latest_time_interaction.append(src_node_interact_times[idx].item() - src_nodes_neighbor_times_list[idx][find_interact_index[0][-1]].item())

        pair_latest_time_interaction = torch.exp(-torch.from_numpy(np.array(src_latest_time_interaction)).to(self.node_raw_features.device) * shrink_coeff)

        return pair_latest_time_interaction

    def time_modeling(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_node_interact_times: np.ndarray,
                      src_nodes_neighbor_ids_list: list, src_nodes_neighbor_times_list: list, time_encoder):

        max_interaction_times = self.max_interaction_times

        padded_time = np.ones((len(src_node_ids), max_interaction_times)).astype(np.longlong) * 1e10

        for idx in range(len(src_node_ids)):
            find_interact = np.where(src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx], src_nodes_neighbor_ids_list[idx], 0)
            find_interact_index = np.nonzero(find_interact)
            if find_interact_index[0].shape[0] == 0: # previous interaction not found
                continue
            else:
                unique_ts = np.unique(src_nodes_neighbor_times_list[idx][find_interact_index[0]])
                find_idx_back = np.concatenate((unique_ts, [src_node_interact_times[idx].item()]))
                find_idx_front = np.concatenate(([0.0], unique_ts))
                time_diff = find_idx_back - find_idx_front
                if time_diff.shape[0] - 1 < max_interaction_times:
                    padded_time[idx][-time_diff.shape[0]+1:] = time_diff[1:]
                else:
                    padded_time[idx][:] = time_diff[-max_interaction_times:]


        padded_time_diff_emb = time_encoder(timestamps=torch.from_numpy(padded_time).float().to(self.node_raw_features.device))

        return padded_time_diff_emb

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.node_raw_features.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size


        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler):
        
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class DyGMambaLP(nn.Module):

    def __init__(
        self,
        ngh_finder,
        n_feat: np.ndarray,
        e_feat: np.ndarray,
        time_feat_dim: int = 100,
        channel_embedding_dim: int = 50,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        drop_out: float = 0.1,
        gamma: float = 0.5,
        max_input_sequence_length: int = 64,
        max_interaction_times: int = 10,
        device: str = 'cpu',
    ):
        super().__init__()
        self._ngh_finder = ngh_finder
        self.device = device
        self.backbone = DyGMamba(
            node_raw_features=n_feat,
            edge_raw_features=e_feat,
            neighbor_sampler=NeighborFinderAdapter(ngh_finder),
            time_feat_dim=time_feat_dim,
            channel_embedding_dim=channel_embedding_dim,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=drop_out,
            gamma=gamma,
            max_input_sequence_length=max_input_sequence_length,
            max_interaction_times=max_interaction_times,
            device=device,
        )
        node_feat_dim = n_feat.shape[1]
        self.link_predictor = MergeLayerTD(
            input_dim1=node_feat_dim,
            input_dim2=node_feat_dim,
            input_dim3=node_feat_dim,
            hidden_dim=node_feat_dim,
            output_dim=1,
        )

    @property
    def ngh_finder(self):
        return self._ngh_finder

    @ngh_finder.setter
    def ngh_finder(self, finder):
        self._ngh_finder = finder
        self.backbone.set_neighbor_sampler(NeighborFinderAdapter(finder))

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        # num_neighbors kept for TGAT API compatibility; DyGMamba uses max_input_sequence_length
        device = self.backbone.node_raw_features.device
        self.backbone.device = device
        self.backbone.neighbor_co_occurrence_encoder.device = device
        src_emb, dst_emb, time_diff_emb = self.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=np.asarray(src_idx_l),
            dst_node_ids=np.asarray(target_idx_l),
            node_interact_times=np.asarray(cut_time_l),
        )
        neg_src_emb, neg_dst_emb, neg_time_diff_emb = self.backbone.compute_src_dst_node_temporal_embeddings(
            src_node_ids=np.asarray(src_idx_l),
            dst_node_ids=np.asarray(background_idx_l),
            node_interact_times=np.asarray(cut_time_l),
        )
        pos_score = self.link_predictor(src_emb, dst_emb, time_diff_emb).squeeze(dim=-1).sigmoid()
        neg_score = self.link_predictor(neg_src_emb, neg_dst_emb, neg_time_diff_emb).squeeze(dim=-1).sigmoid()
        return pos_score, neg_score


class NIFEncoder(nn.Module):

    def __init__(self, nif_feat_dim: int, device: str = 'cpu'):

        super(NIFEncoder, self).__init__()

        self.nif_feat_dim = nif_feat_dim
        self.device = device

        self.nif_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.nif_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.nif_feat_dim, out_features=self.nif_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                src_nodes_neighbor_ids: np.ndarray, dst_nodes_neighbor_ids: np.ndarray):

        # two lists to store the appearances of source and destination nodes
        src_nodes_appearances, dst_nodes_appearances = [], []
        # src_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for i in range(len(src_node_ids)):
            src_node_id = src_node_ids[i]
            dst_node_id = dst_node_ids[i]
            src_node_neighbor_ids = src_nodes_neighbor_ids[i]
            dst_node_neighbor_ids = dst_nodes_neighbor_ids[i]

            # Calculate unique keys and counts for source and destination
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_node_neighbor_ids, return_inverse=True,
                                                                         return_counts=True)

            # Create mappings from node IDs to their counts
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # Adjust counts specifically for the cases where src_node_id appears in dst's neighbors and vice versa
            if src_node_id in dst_mapping_dict:
                src_count_in_dst = dst_mapping_dict[src_node_id]
                src_mapping_dict[src_node_id] = src_count_in_dst
                dst_mapping_dict[src_node_id] = src_count_in_dst
            if dst_node_id in src_mapping_dict:
                dst_count_in_src = src_mapping_dict[dst_node_id]
                src_mapping_dict[dst_node_id] = dst_count_in_src
                dst_mapping_dict[dst_node_id] = dst_count_in_src

            # Calculate appearances in each other's lists
            src_node_neighbor_counts_in_dst = torch.tensor(
                [dst_mapping_dict.get(neighbor_id, 0) for neighbor_id in src_node_neighbor_ids]).float().to(self.device)
            dst_node_neighbor_counts_in_src = torch.tensor(
                [src_mapping_dict.get(neighbor_id, 0) for neighbor_id in dst_node_neighbor_ids]).float().to(self.device)

            # Stack counts to get a two-column tensor for each node list
            src_nodes_appearances.append(torch.stack(
                [torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device),
                 src_node_neighbor_counts_in_dst], dim=1))
            dst_nodes_appearances.append(torch.stack([dst_node_neighbor_counts_in_src,
                                                      torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(
                                                          self.device)], dim=1))

        # Stack to form batch tensors
        src_nodes_appearances = torch.stack(src_nodes_appearances, dim=0)
        dst_nodes_appearances = torch.stack(dst_nodes_appearances, dim=0)

        return src_nodes_appearances, dst_nodes_appearances

    def forward(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, src_nodes_neighbor_ids: np.ndarray,
                dst_nodes_neighbor_ids: np.ndarray):
       
        src_nodes_appearances, dst_nodes_appearances = self.count_nodes_appearances(src_node_ids=src_node_ids,
                                                                                    dst_node_ids=dst_node_ids,
                                                                                    src_nodes_neighbor_ids=src_nodes_neighbor_ids,
                                                                                    dst_nodes_neighbor_ids=dst_nodes_neighbor_ids)


        src_nodes_nif_features = (src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = (dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        src_nodes_nif_features = self.nif_encode_layer(src_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        dst_nodes_nif_features = self.nif_encode_layer(dst_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        return src_nodes_nif_features, dst_nodes_nif_features


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return self.ffn(x)
