import numpy as np
import torch
import torch.nn as nn

from utils.dygmamba_module import (
    DyGMamba,
    FeedForwardNet,
    MergeLayerTD,
    NeighborFinderAdapter,
    NIFEncoder,
    TimeEncoder,
)
from utils.sl_rnn_module import SLSeq


class DyGSLRNN(DyGMamba):
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
        nn.Module.__init__(self)

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
        self.mamba = nn.ModuleList([SLSeq(d_model=d_node, **sl_kwargs) for _ in range(self.num_layers)])
        self.mamba_t_diff = nn.ModuleList([SLSeq(d_model=d_time, **sl_kwargs) for _ in range(self.num_layers)])

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
