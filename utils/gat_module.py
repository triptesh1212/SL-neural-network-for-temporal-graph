import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from utils.module import MergeLayer


class GATAttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # a = [a_l || a_r]; score = a_l^T W h_i + a_r^T W h_j
        self.a_l = nn.Linear(out_dim, 1, bias=False)
        self.a_r = nn.Linear(out_dim, 1, bias=False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l.weight)
        nn.init.xavier_uniform_(self.a_r.weight)

    def forward(self, src, seq, mask):
        src = F.dropout(src, p=self.dropout, training=self.training)
        seq = F.dropout(seq, p=self.dropout, training=self.training)

        Wh_src = self.W(src)                       # [B, D_out]
        Wh_seq = self.W(seq)                       # [B, N, D_out]

        # e_ij = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j)
        score_src = self.a_l(Wh_src)               # [B, 1]
        score_ngh = self.a_r(Wh_seq).squeeze(-1)   # [B, N]
        logits = score_src + score_ngh             # [B, N]
        logits = F.leaky_relu(logits, negative_slope=self.negative_slope)

        logits = logits.masked_fill(mask, float('-inf'))
        attn = torch.softmax(logits, dim=-1)       # [B, N]
        # padded rows become NaN after softmax; zero them out
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.bmm(attn.unsqueeze(1), Wh_seq).squeeze(1)  # [B, D_out]
        return out, attn


class GATMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, n_head=2, dropout=0.1, negative_slope=0.2):
        super().__init__()
        assert out_dim % n_head == 0
        self.n_head = n_head
        head_dim = out_dim // n_head

        self.heads = nn.ModuleList([
            GATAttentionHead(in_dim, head_dim, dropout=dropout, negative_slope=negative_slope)
            for _ in range(n_head)
        ])
        self.fc = nn.Linear(out_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, src, seq, mask):
        head_outs, attns = [], []
        for head in self.heads:
            out, attn = head(src, seq, mask)
            head_outs.append(out)
            attns.append(attn)
        # Concatenate heads (as in GAT hidden layers)
        concat = torch.cat(head_outs, dim=-1)      # [B, out_dim]
        out = self.act(self.fc(self.dropout(concat)))
        attn = torch.stack(attns, dim=0).mean(dim=0)
        return out, attn


class GATAttnModel(nn.Module):
    """GAT aggregation over temporal neighbours (no time encoding)"""

    def __init__(self, feat_dim, edge_dim, n_head=2, drop_out=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.edge_dim = edge_dim

        # Project [node || edge] neighbour features to feat_dim
        self.ngh_proj = nn.Linear(feat_dim + edge_dim, feat_dim, bias=False)
        nn.init.xavier_uniform_(self.ngh_proj.weight)

        self.multi_head = GATMultiHeadAttention(
            in_dim=feat_dim, out_dim=feat_dim, n_head=n_head, dropout=drop_out
        )
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, seq, seq_e, mask):
        ngh = self.ngh_proj(torch.cat([seq, seq_e], dim=-1))  # [B, N, D]
        output, attn = self.multi_head(src, ngh, mask)
        output = self.merger(output, src)
        return output, attn


class GATAN(nn.Module):
    def __init__(
        self,
        ngh_finder,
        n_feat,
        e_feat,
        num_layers=2,
        n_head=2,
        null_idx=0,
        drop_out=0.1,
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

        self.logger.info('Aggregation uses GAT attention')
        self.attn_model_list = nn.ModuleList([
            GATAttnModel(self.feat_dim, self.feat_dim, n_head=n_head, drop_out=drop_out)
            for _ in range(num_layers)
        ])

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

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

        src_ngh_node_batch_flat = src_ngh_node_batch.flatten()
        src_ngh_t_batch_flat = src_ngh_t_batch.flatten()
        src_ngh_node_conv_feat = self.tem_conv(
            src_ngh_node_batch_flat,
            src_ngh_t_batch_flat,
            curr_layers=curr_layers - 1,
            num_neighbors=num_neighbors,
        )
        src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
        src_ngh_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

        mask = src_ngh_node_batch_th == 0
        attn_m = self.attn_model_list[curr_layers - 1]
        local, _ = attn_m(src_node_conv_feat, src_ngh_feat, src_ngh_edge_feat, mask)
        return local
