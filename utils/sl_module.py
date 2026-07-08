import logging

import numpy as np
import torch
import torch.nn as nn

from utils.module import (
    TimeEncode,
    PosEncode,
    EmptyEncode,
    MergeLayer,
    MultiHeadAttention,
    MapBasedMultiHeadAttention,
)


def imex_sl_step(z, coupling, dt, alpha, omega, beta, gamma, tol=1e-5, max_iter=20):
    z_tilde = z + dt * coupling

    r_tilde = z_tilde.abs().clamp(min=1e-8)
    phi_tilde = torch.angle(z_tilde)

    r_new = r_tilde.clone()
    for _ in range(max_iter):
        residual = r_new - r_tilde - dt * (alpha * r_new - beta * r_new ** 3)
        deriv = 1.0 - dt * (alpha - 3.0 * beta * r_new ** 2)
        update = residual / deriv.clamp(min=1e-8)
        r_new = r_new - update
        if torch.max(torch.abs(update)) < tol:
            break

    phi_new = phi_tilde + dt * (omega + gamma * r_new ** 2)
    return torch.polar(r_new, phi_new)


def sl_exact_step(h, zeta, nu, dt):
    h = torch.as_tensor(h)
    if not torch.is_complex(h):
        h = h.to(torch.complex64)
    zeta = torch.as_tensor(zeta, dtype=h.dtype, device=h.device)
    nu = torch.as_tensor(nu, dtype=h.dtype, device=h.device)
    zr, nr = zeta.real, nu.real
    dt = torch.as_tensor(dt, dtype=zr.dtype, device=h.device)

    propagator = torch.exp(zeta * dt)
    absh2 = h.real ** 2 + h.imag ** 2
    c = (nr / zr) * torch.expm1(2.0 * zr * dt)
    d = 1.0 + c * absh2
    saturation = torch.exp((-nu / (2.0 * nr)) * torch.log(d))
    return propagator * saturation * h


def exact_sl_step(z, coupling, dt, alpha, omega, beta, gamma):
    z_tilde = z + dt * coupling
    zeta = torch.complex(alpha, omega)
    nu = torch.complex(beta, -gamma)
    return sl_exact_step(z_tilde, zeta, nu, dt)


class SLAttnModel(nn.Module):

    def __init__(
        self,
        feat_dim,
        edge_dim,
        time_dim,
        attn_mode='prod',
        n_head=2,
        drop_out=0.1,
        dt=1.0,
        sl_alpha=0.04,
        sl_omega=0.5,
        sl_beta=1.0,
        sl_gamma=0.0,
        coupling_strength=1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.edge_in_dim = feat_dim + edge_dim + time_dim
        self.model_dim = self.edge_in_dim
        self.dt = dt
        self.coupling_strength = coupling_strength

        self.sl_alpha = nn.Parameter(torch.tensor(float(sl_alpha)))
        self.sl_omega = nn.Parameter(torch.tensor(float(sl_omega)))
        self.sl_beta = nn.Parameter(torch.tensor(float(sl_beta)))
        self.sl_gamma = nn.Parameter(torch.tensor(float(sl_gamma)))

        assert self.model_dim % n_head == 0
        self.logger = logging.getLogger(__name__)

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out,
            )
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out,
            )
        else:
            raise ValueError('attn_mode can only be prod or map')

        self.complex_enc = nn.Linear(feat_dim, feat_dim, dtype=torch.cfloat)
        self.h_proj = nn.Linear(self.model_dim, feat_dim)
        self.coupling_ffn = MergeLayer(feat_dim, 2 * feat_dim, feat_dim, 2 * feat_dim)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def _temporal_attention(self, src, src_t, seq, seq_t, seq_e, mask):
        src_ext = torch.unsqueeze(src, dim=1)
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)
        k = torch.cat([seq, seq_e, seq_t], dim=2)

        mask = torch.unsqueeze(mask, dim=2).permute([0, 2, 1])
        h_v, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)
        h_v = h_v.squeeze(1)

        batch_size = q.size(0)
        n_head = self.multi_head_target.n_head
        if attn.dim() == 3 and attn.size(0) == n_head * batch_size:
            attn = attn.view(n_head, batch_size, attn.size(1), attn.size(2)).mean(dim=0)
        attn = attn.squeeze(1)
        return h_v, attn

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        h_v, attn = self._temporal_attention(src, src_t, seq, seq_t, seq_e, mask)

        z_v = self.complex_enc(src.to(torch.cfloat))

        # F_θ^(l)_v = FFN(h_v(t) || z_v^(l)) with full complex state
        h_v_feat = self.h_proj(h_v)
        z_v_ri = torch.cat([z_v.real, z_v.imag], dim=-1)
        coupling_ri = self.coupling_ffn(h_v_feat, z_v_ri)
        coupling = torch.view_as_complex(
            coupling_ri.reshape(*coupling_ri.shape[:-1], self.feat_dim, 2)
        )
        coupling = self.coupling_strength * coupling

        # z_out = imex_sl_step(
        #     z_v,
        #     coupling,
        #     self.dt,
        #     self.sl_alpha,
        #     self.sl_omega,
        #     self.sl_beta,
        #     self.sl_gamma,
        # )

        z_out = exact_sl_step(
            z_v,
            coupling,
            self.dt,
            self.sl_alpha,
            self.sl_omega,
            self.sl_beta,
            self.sl_gamma,
        )

        output = self.merger(z_out.real, src)
        return output, attn


class SLAttnModelNeighborDiff(nn.Module):

    def __init__(
        self,
        feat_dim,
        edge_dim,
        time_dim,
        attn_mode='prod',
        n_head=2,
        drop_out=0.1,
        dt=1.0,
        sl_alpha=0.04,
        sl_omega=0.5,
        sl_beta=1.0,
        sl_gamma=0.0,
        coupling_strength=1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.edge_in_dim = feat_dim + edge_dim + time_dim
        self.model_dim = self.edge_in_dim
        self.dt = dt
        self.coupling_strength = coupling_strength

        self.sl_alpha = nn.Parameter(torch.tensor(float(sl_alpha)))
        self.sl_omega = nn.Parameter(torch.tensor(float(sl_omega)))
        self.sl_beta = nn.Parameter(torch.tensor(float(sl_beta)))
        self.sl_gamma = nn.Parameter(torch.tensor(float(sl_gamma)))

        assert self.model_dim % n_head == 0
        self.logger = logging.getLogger(__name__)

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out,
            )
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out,
            )
        else:
            raise ValueError('attn_mode can only be prod or map')

        self.complex_enc = nn.Linear(feat_dim, feat_dim, dtype=torch.cfloat)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def _temporal_attention(self, src, src_t, seq, seq_t, seq_e, mask):
        src_ext = torch.unsqueeze(src, dim=1)
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)
        k = torch.cat([seq, seq_e, seq_t], dim=2)

        mask = torch.unsqueeze(mask, dim=2).permute([0, 2, 1])
        _, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)

        batch_size = q.size(0)
        n_head = self.multi_head_target.n_head
        if attn.dim() == 3 and attn.size(0) == n_head * batch_size:
            attn = attn.view(n_head, batch_size, attn.size(1), attn.size(2)).mean(dim=0)
        attn = attn.squeeze(1)
        return attn

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        attn = self._temporal_attention(src, src_t, seq, seq_t, seq_e, mask)

        z_v = self.complex_enc(src.to(torch.cfloat))
        z_u = self.complex_enc(seq.to(torch.cfloat))

        # F_theta^(l)_v = sum_u A_vu * (Z_u - Z_v)
        diff = z_u - z_v.unsqueeze(1)
        coupling = (attn.unsqueeze(-1) * diff).sum(dim=1)
        coupling = self.coupling_strength * coupling

        z_out = exact_sl_step(
            z_v,
            coupling,
            self.dt,
            self.sl_alpha,
            self.sl_omega,
            self.sl_beta,
            self.sl_gamma,
        )

        output = self.merger(z_out.real, src)
        return output, attn


class SLTGAN(nn.Module):
    def __init__(
        self,
        ngh_finder,
        n_feat,
        e_feat,
        attn_mode='prod',
        use_time='time',
        num_layers=2,
        n_head=2,
        null_idx=0,
        drop_out=0.1,
        seq_len=None,
        dt=1.0,
        sl_alpha=0.04,
        sl_omega=0.5,
        sl_beta=1.0,
        sl_gamma=0.0,
        coupling_strength=1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]

        self.attn_model_list = nn.ModuleList([
            SLAttnModel(
                self.feat_dim,
                self.feat_dim,
                self.feat_dim,
                attn_mode=attn_mode,
                n_head=n_head,
                drop_out=drop_out,
                dt=dt,
                sl_alpha=sl_alpha,
                sl_omega=sl_omega,
                sl_beta=sl_beta,
                sl_gamma=sl_gamma,
                coupling_strength=coupling_strength,
            )
            for _ in range(num_layers)
        ])

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif use_time == 'pos':
            assert seq_len is not None
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        return self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert curr_layers >= 0

        device = self.n_feat_th.device
        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat

        src_node_conv_feat = self.tem_conv(
            src_idx_l, cut_time_l, curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors
        )

        src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
        src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)

        src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_node_conv_feat = self.tem_conv(
            src_ngh_node_batch_flat,
            src_ngh_t_batch_flat,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors,
        )
        src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
        src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

        mask = src_ngh_node_batch_th == 0
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(
            src_node_conv_feat,
            src_node_t_embed,
            src_ngh_feat,
            src_ngh_t_embed,
            src_ngn_edge_feat,
            mask,
        )
        return local


class SLTGANNeighborDiff(nn.Module):
    def __init__(
        self,
        ngh_finder,
        n_feat,
        e_feat,
        attn_mode='prod',
        use_time='time',
        num_layers=2,
        n_head=2,
        null_idx=0,
        drop_out=0.1,
        seq_len=None,
        dt=1.0,
        sl_alpha=0.04,
        sl_omega=0.5,
        sl_beta=1.0,
        sl_gamma=0.0,
        coupling_strength=1.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]

        self.attn_model_list = nn.ModuleList([
            SLAttnModelNeighborDiff(
                self.feat_dim,
                self.feat_dim,
                self.feat_dim,
                attn_mode=attn_mode,
                n_head=n_head,
                drop_out=drop_out,
                dt=dt,
                sl_alpha=sl_alpha,
                sl_omega=sl_omega,
                sl_beta=sl_beta,
                sl_gamma=sl_gamma,
                coupling_strength=coupling_strength,
            )
            for _ in range(num_layers)
        ])

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.feat_dim)
        elif use_time == 'pos':
            assert seq_len is not None
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.feat_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.feat_dim)
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        return self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert curr_layers >= 0

        device = self.n_feat_th.device
        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)

        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat

        src_node_conv_feat = self.tem_conv(
            src_idx_l, cut_time_l, curr_layers=curr_layers - 1, num_neighbors=num_neighbors
        )

        src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors
        )

        src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
        src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)

        src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
        src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)

        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_node_conv_feat = self.tem_conv(
            src_ngh_node_batch_flat,
            src_ngh_t_batch_flat,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors,
        )
        src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

        src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
        src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

        mask = src_ngh_node_batch_th == 0
        attn_m = self.attn_model_list[curr_layers - 1]

        local, _ = attn_m(
            src_node_conv_feat,
            src_node_t_embed,
            src_ngh_feat,
            src_ngh_t_embed,
            src_ngn_edge_feat,
            mask,
        )
        return local
