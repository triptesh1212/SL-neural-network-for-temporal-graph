"""DyG-SLRNN: DyG-style temporal link prediction with Stuart-Landau sequence mixers."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sl_rnn_module import SLSeq
from utils.utils import (
    FeedForwardNet,
    MergeLayerTD,
    NeighborFinderAdapter,
    NIFEncoder,
    TimeEncoder,
)


class DyGSLRNN(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler,
        time_feat_dim: int,
        channel_embedding_dim: int,
        patch_size: int = 1,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        gamma: float = 0.5,
        max_input_sequence_length: int = 512,
        max_interaction_times: int = 10,
        device: str = 'cpu',
        sl_dt: float = 1.0,
        sl_p: int = 2,
        sl_tol: float = 1e-5,
        zeta_real: float = 0.04,
        zeta_imag: float = 0.5,
        nu_real: float = 1.0,
        nu_imag: float = 0.0,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()

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
        self.neighbor_co_occurrence_encoder = NIFEncoder(
            nif_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(
                in_features=self.patch_size * self.node_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            'edge': nn.Linear(
                in_features=self.patch_size * self.edge_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            'time': nn.Linear(
                in_features=self.patch_size * self.time_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
            'neighbor_co_occurrence': nn.Linear(
                in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim,
                out_features=self.channel_embedding_dim,
                bias=True,
            ),
        })

        self.num_channels = 4
        feature_expansion_size = 2
        d_node = self.num_channels * self.channel_embedding_dim // feature_expansion_size
        d_time = int(self.gamma * self.channel_embedding_dim)

        self.output_layer = nn.Linear(in_features=d_node, out_features=self.node_feat_dim, bias=True)
        self.output_layer_t_diff = nn.Linear(in_features=d_time, out_features=self.node_feat_dim, bias=True)

        sl_kwargs = dict(
            dt=sl_dt,
            p=sl_p,
            tol=sl_tol,
            zeta_real=zeta_real,
            zeta_imag=zeta_imag,
            nu_real=nu_real,
            nu_imag=nu_imag,
            leaky_relu_slope=leaky_relu_slope,
            use_hid_enc=True,
        )
        self.sl_layers = nn.ModuleList([SLSeq(d_model=d_node, **sl_kwargs) for _ in range(self.num_layers)])
        self.sl_t_diff = nn.ModuleList([SLSeq(d_model=d_time, **sl_kwargs) for _ in range(self.num_layers)])

        self.projection_layer_t_diff = nn.Linear(
            in_features=self.time_feat_dim, out_features=d_time, bias=True
        )
        self.projection_layer_t_diff_up = nn.Linear(
            in_features=d_time, out_features=d_node, bias=True
        )

        self.weightagg = nn.Linear(d_node, 1)
        self.reduce_layer = nn.Linear(
            self.num_channels * self.channel_embedding_dim, d_node
        )
        self.channel_norm = nn.LayerNorm(d_node)
        self.channel_feedforward = FeedForwardNet(
            input_dim=d_node, dim_expansion_factor=4, dropout=self.dropout
        )
        self.neighbor_selection_layer = nn.Linear(d_node, d_node)

    def compute_src_dst_node_temporal_embeddings(
        self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray
    ):
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=src_node_ids, node_interact_times=node_interact_times
            )

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(
                node_ids=dst_node_ids, node_interact_times=node_interact_times
            )

        padded_time_diff_emb = self.time_modeling(
            src_node_ids, dst_node_ids, node_interact_times,
            src_nodes_neighbor_ids_list, src_nodes_neighbor_times_list, self.time_encoder
        )

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=src_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                nodes_edge_ids_list=src_nodes_edge_ids_list,
                nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length,
            )

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(
                node_ids=dst_node_ids,
                node_interact_times=node_interact_times,
                nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                nodes_edge_ids_list=dst_nodes_edge_ids_list,
                nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                patch_size=self.patch_size,
                max_input_sequence_length=self.max_input_sequence_length,
            )

        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(
                src_node_ids=src_node_ids,
                dst_node_ids=dst_node_ids,
                src_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            )

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=src_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
                padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                time_encoder=self.time_encoder,
            )

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
            src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=src_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
            dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(
                padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                padded_nodes_edge_raw_features=dst_padded_nodes_edge_raw_features,
                padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                patch_size=self.patch_size,
            )

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](
            src_patches_nodes_neighbor_node_raw_features
        )
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](
            src_patches_nodes_edge_raw_features
        )
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](
            src_patches_nodes_neighbor_time_features
        )
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            src_patches_nodes_neighbor_co_occurrence_features
        )

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](
            dst_patches_nodes_neighbor_node_raw_features
        )
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](
            dst_patches_nodes_edge_raw_features
        )
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](
            dst_patches_nodes_neighbor_time_features
        )
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](
            dst_patches_nodes_neighbor_co_occurrence_features
        )

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        src_patches_data = [
            src_patches_nodes_neighbor_node_raw_features,
            src_patches_nodes_edge_raw_features,
            src_patches_nodes_neighbor_time_features,
            src_patches_nodes_neighbor_co_occurrence_features,
        ]
        dst_patches_data = [
            dst_patches_nodes_neighbor_node_raw_features,
            dst_patches_nodes_edge_raw_features,
            dst_patches_nodes_neighbor_time_features,
            dst_patches_nodes_neighbor_co_occurrence_features,
        ]
        src_patches_data = torch.stack(src_patches_data, dim=2)
        dst_patches_data = torch.stack(dst_patches_data, dim=2)
        src_patches_data = src_patches_data.reshape(
            batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim
        )
        dst_patches_data = dst_patches_data.reshape(
            batch_size, dst_num_patches, self.num_channels * self.channel_embedding_dim
        )

        src_patches_data = self.reduce_layer(src_patches_data)
        dst_patches_data = self.reduce_layer(dst_patches_data)

        for sl_layer in self.sl_layers:
            src_patches_data = sl_layer(src_patches_data) + src_patches_data
            dst_patches_data = sl_layer(dst_patches_data) + dst_patches_data
            src_patches_data = self.channel_norm(src_patches_data)
            dst_patches_data = self.channel_norm(dst_patches_data)
            src_patches_data = self.channel_feedforward(src_patches_data) + src_patches_data
            dst_patches_data = self.channel_feedforward(dst_patches_data) + dst_patches_data

        padded_time_diff_emb = self.projection_layer_t_diff(padded_time_diff_emb)
        for sl_t in self.sl_t_diff:
            padded_time_diff_emb = sl_t(padded_time_diff_emb) + padded_time_diff_emb

        src_weight = self.weightagg(src_patches_data).transpose(1, 2)
        dst_weight = self.weightagg(dst_patches_data).transpose(1, 2)

        src_patches_data_ = src_weight.matmul(src_patches_data).squeeze(dim=1)
        dst_patches_data_ = dst_weight.matmul(dst_patches_data).squeeze(dim=1)

        time_diff_emb = torch.mean(padded_time_diff_emb, dim=1)
        time_diff_emb_ = self.projection_layer_t_diff_up(time_diff_emb)

        src_selection_param = (self.neighbor_selection_layer(dst_patches_data_) * time_diff_emb_).unsqueeze(1)
        dst_selection_param = (self.neighbor_selection_layer(src_patches_data_) * time_diff_emb_).unsqueeze(1)

        src_patches_data = torch.sum(
            src_patches_data
            * F.softmax(torch.sum(src_selection_param * src_patches_data, dim=2), dim=1).unsqueeze(2),
            dim=1,
        )
        dst_patches_data = torch.sum(
            dst_patches_data
            * F.softmax(torch.sum(dst_selection_param * dst_patches_data, dim=2), dim=1).unsqueeze(2),
            dim=1,
        )

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)
        time_diff_emb = self.output_layer_t_diff(time_diff_emb)
        return src_node_embeddings, dst_node_embeddings, time_diff_emb

    def pad_sequences(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids_list: list,
        nodes_edge_ids_list: list,
        nodes_neighbor_times_list: list,
        patch_size: int = 1,
        max_input_sequence_length: int = 256,
    ):
        assert max_input_sequence_length - 1 > 0, \
            'Maximal number of neighbors for each node should be greater than 1!'

        max_seq_length = max_input_sequence_length
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(
                nodes_neighbor_times_list[idx]
            )
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]

        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += patch_size - max_seq_length % patch_size
        assert max_seq_length % patch_size == 0

        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, -1] = node_ids[idx]
            padded_nodes_edge_ids[idx, -1] = 0
            padded_nodes_neighbor_times[idx, -1] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, -len(nodes_neighbor_ids_list[idx]) - 1:-1] = \
                    nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, -len(nodes_edge_ids_list[idx]) - 1:-1] = \
                    nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, -len(nodes_neighbor_times_list[idx]) - 1:-1] = \
                    nodes_neighbor_times_list[idx]

        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def time_modeling(
        self,
        src_node_ids: np.ndarray,
        dst_node_ids: np.ndarray,
        src_node_interact_times: np.ndarray,
        src_nodes_neighbor_ids_list: list,
        src_nodes_neighbor_times_list: list,
        time_encoder,
    ):
        max_interaction_times = self.max_interaction_times
        padded_time = np.ones((len(src_node_ids), max_interaction_times)).astype(np.longlong) * 1e10

        for idx in range(len(src_node_ids)):
            find_interact = np.where(
                src_nodes_neighbor_ids_list[idx] == dst_node_ids[idx],
                src_nodes_neighbor_ids_list[idx],
                0,
            )
            find_interact_index = np.nonzero(find_interact)
            if find_interact_index[0].shape[0] == 0:
                continue
            unique_ts = np.unique(src_nodes_neighbor_times_list[idx][find_interact_index[0]])
            find_idx_back = np.concatenate((unique_ts, [src_node_interact_times[idx].item()]))
            find_idx_front = np.concatenate(([0.0], unique_ts))
            time_diff = find_idx_back - find_idx_front
            if time_diff.shape[0] - 1 < max_interaction_times:
                padded_time[idx][-time_diff.shape[0] + 1:] = time_diff[1:]
            else:
                padded_time[idx][:] = time_diff[-max_interaction_times:]

        padded_time_diff_emb = time_encoder(
            timestamps=torch.from_numpy(padded_time).float().to(self.node_raw_features.device)
        )
        return padded_time_diff_emb

    def get_features(
        self,
        node_interact_times: np.ndarray,
        padded_nodes_neighbor_ids: np.ndarray,
        padded_nodes_edge_ids: np.ndarray,
        padded_nodes_neighbor_times: np.ndarray,
        time_encoder: TimeEncoder,
    ):
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            torch.from_numpy(padded_nodes_neighbor_ids)
        ]
        padded_nodes_edge_raw_features = self.edge_raw_features[
            torch.from_numpy(padded_nodes_edge_ids)
        ]
        padded_nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(
                node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times
            ).float().to(self.node_raw_features.device)
        )
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0
        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def get_patches(
        self,
        padded_nodes_neighbor_node_raw_features: torch.Tensor,
        padded_nodes_edge_raw_features: torch.Tensor,
        padded_nodes_neighbor_time_features: torch.Tensor,
        padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None,
        patch_size: int = 1,
    ):
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        patches_nodes_neighbor_node_raw_features = []
        patches_nodes_edge_raw_features = []
        patches_nodes_neighbor_time_features = []
        patches_nodes_neighbor_co_occurrence_features = []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(
                padded_nodes_neighbor_node_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_edge_raw_features.append(
                padded_nodes_edge_raw_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_time_features.append(
                padded_nodes_neighbor_time_features[:, start_idx:end_idx, :]
            )
            patches_nodes_neighbor_co_occurrence_features.append(
                padded_nodes_neighbor_co_occurrence_features[:, start_idx:end_idx, :]
            )

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        patches_nodes_neighbor_node_raw_features = torch.stack(
            patches_nodes_neighbor_node_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(
            patches_nodes_edge_raw_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(
            patches_nodes_neighbor_time_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(
            patches_nodes_neighbor_co_occurrence_features, dim=1
        ).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return (
            patches_nodes_neighbor_node_raw_features,
            patches_nodes_edge_raw_features,
            patches_nodes_neighbor_time_features,
            patches_nodes_neighbor_co_occurrence_features,
        )

    def set_neighbor_sampler(self, neighbor_sampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class DyGSLRNNLP(nn.Module):
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
        sl_dt: float = 1.0,
        sl_p: int = 2,
        sl_tol: float = 1e-5,
        zeta_real: float = 0.04,
        zeta_imag: float = 0.5,
        nu_real: float = 1.0,
        nu_imag: float = 0.0,
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self._ngh_finder = ngh_finder
        self.device = device
        self.backbone = DyGSLRNN(
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
            sl_dt=sl_dt,
            sl_p=sl_p,
            sl_tol=sl_tol,
            zeta_real=zeta_real,
            zeta_imag=zeta_imag,
            nu_real=nu_real,
            nu_imag=nu_imag,
            leaky_relu_slope=leaky_relu_slope,
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
