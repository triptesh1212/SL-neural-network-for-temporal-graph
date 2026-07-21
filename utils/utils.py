import numpy as np
import torch
import torch.nn as nn

class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

"""DyGMamba-style early stopping"""
class EarlyStopMonitor2(object):
    def __init__(self, patience=20):
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.best_epoch = 0
        self.early_stop = False

    def step(self, metrics, epoch):
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value >= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value <= self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)

        # all computed metrics are better than (or equal to) the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            improved = False

        return self.early_stop, improved


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

"""DyGMamba-style random negative edge sampler"""
class NegativeEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

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
