"""SL-TGAT: Stuart-Landau Temporal Graph Attention for link prediction."""
import math
import logging
import time
import random
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from utils.sl_module import SLTGAN
from utils.graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

parser = argparse.ArgumentParser('SL-TGAT experiments on temporal link prediction')
parser.add_argument('-d', '--data', type=str, default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=5, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], default='time')
parser.add_argument('--uniform', action='store_true', help='uniform temporal neighbor sampling')
# Stuart-Landau parameters (main-algo.pdf page 2: F_theta = FFN(h_v || z_v))
parser.add_argument('--dt', type=float, default=1.0, help='IMEX time step')
parser.add_argument('--sl_alpha', type=float, default=0.04, help='Hopf parameter (real part)')
parser.add_argument('--sl_omega', type=float, default=0.5, help='natural frequency')
parser.add_argument('--sl_beta', type=float, default=1.0, help='amplitude stabilization')
parser.add_argument('--sl_gamma', type=float, default=0.0, help='phase shift parameter')
parser.add_argument('--coupling_strength', type=float, default=1.0, help='scale on F_theta')

try:
    args = parser.parse_args()
except Exception:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr

tag = f'sl-tgat-{args.attn_mode}'
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{tag}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{tag}-{args.data}-{epoch}.pth'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, model, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        model = model.eval()
        test_batch_size = 30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / test_batch_size)
        for k in range(num_test_batch):
            s_idx = k * test_batch_size
            e_idx = min(num_test_instance - 1, s_idx + test_batch_size)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            size = len(src_l_cut)
            _, dst_l_fake = sampler.sample(size)
            pos_prob, neg_prob = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            pred_score = np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)


g_df = pd.read_csv('./data/ml_{}.csv'.format(DATA))
e_feat = np.load('./data/ml_{}.npy'.format(DATA))
n_feat = np.load('./data/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_idx = max(src_l.max(), dst_l.max())
random.seed(2020)

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

candidate_nodes = list(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])))
mask_node_set = set(random.sample(candidate_nodes, min(int(0.1 * num_total_unique_nodes), len(candidate_nodes))))
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

train_node_set = set(train_src_l).union(train_dst_l)
assert len(train_node_set - mask_node_set) == len(train_node_set)
new_node_set = total_node_set - train_node_set

valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge

val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]

nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(GPU))
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = SLTGAN(
    train_ngh_finder,
    n_feat,
    e_feat,
    num_layers=NUM_LAYER,
    use_time=USE_TIME,
    attn_mode=ATTN_MODE,
    seq_len=SEQ_LEN,
    n_head=NUM_HEADS,
    drop_out=DROP_OUT,
    dt=args.dt,
    sl_alpha=args.sl_alpha,
    sl_omega=args.sl_omega,
    sl_beta=args.sl_beta,
    sl_gamma=args.sl_gamma,
    coupling_strength=args.coupling_strength,
)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
model = model.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

early_stopper = EarlyStopMonitor()
for epoch in range(NUM_EPOCH):
    model.ngh_finder = train_ngh_finder
    acc, ap, auc, m_loss = [], [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        size = len(src_l_cut)
        _, dst_l_fake = train_rand_sampler.sample(size)

        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        model.train()
        pos_prob, neg_prob = model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
        loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))

    model.ngh_finder = full_ngh_finder
    val_acc, val_ap, _, val_auc = eval_one_epoch('val old', model, val_rand_sampler, val_src_l, val_dst_l, val_ts_l, val_label_l)
    nn_val_acc, nn_val_ap, _, nn_val_auc = eval_one_epoch('val new', model, val_rand_sampler, nn_val_src_l, nn_val_dst_l, nn_val_ts_l, nn_val_label_l)

    logger.info('epoch: {}'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
    logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))

    if early_stopper.early_stop_check(val_ap):
        logger.info('Early stopping at epoch {}'.format(epoch))
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        break
    torch.save(model.state_dict(), get_checkpoint_path(epoch))

model.ngh_finder = full_ngh_finder
test_acc, test_ap, _, test_auc = eval_one_epoch('test old', model, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l)
nn_test_acc, nn_test_ap, _, nn_test_auc = eval_one_epoch('test new', model, nn_test_rand_sampler, nn_test_src_l, nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

logger.info('Test old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
logger.info('Test new nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

torch.save(model.state_dict(), MODEL_SAVE_PATH)
logger.info('SL-TGAT model saved to {}'.format(MODEL_SAVE_PATH))
