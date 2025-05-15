"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import pandas as pd
import torch
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from dhg import Hypergraph
from dhg.data import CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationPubmed, CocitationCiteseer
from data import MultilayerHypergraph
from data import CoauthorshipCombinate, CocitationCombinate
from data.bio_data import DrugBank, DisGenet

from data.convert_to_hg import dhg_convert_xgi, other_hypergraph_convert_dhg, nxg_convert_g
from utils.xgi_hypergraph_utils import *


def load_data(args):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats)
    else:
        data = load_data_lp(args.dataset, args.use_feats)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    return data


def process(adj, features, normalize_adj, normalize_feats):
    if isinstance(features, tuple):
        # 使用稀疏矩阵，而不是元组
        features = sp.csr_matrix((features[1], (features[0][:, 0], features[0][:, 1])), shape=features[2])

    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # Convert rowsum to a floating point type to avoid the ValueError
    rowsum = rowsum.astype(np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

from scipy.sparse import coo_matrix


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量。"""

    # 检查sparse_mx是否已经是scipy.sparse矩阵类型
    if not isinstance(sparse_mx, (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix)):
        # 如果不是，将numpy矩阵转换为COO格式的scipy稀疏矩阵
        sparse_mx = coo_matrix(sparse_mx)

    # 现在可以安全地转换为COO格式
    sparse_mx = sparse_mx.tocoo()

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_data_lp(dataset, use_feats):
    if dataset in ['DrugBank', "DisGenet"]:
        adj, features = load_single_bio_data(dataset, use_feats)
    elif dataset == 'biodata':
        adj, features = load_bio_data(dataset, use_feats)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))

    data = {'adj_train': adj, 'features': features}
    return data



def load_data_nc(dataset, use_feats):
    if dataset in ['CoauthorshipCora', 'CoauthorshipDBLP', 'CocitationCora', 'CocitationCiteseer', 'CocitationPubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_dhg_data(dataset, use_feats)
    elif dataset in ['CoauthorshipCombinate', 'CocitationCombinate']:
        adj, features, labels, idx_train, idx_val, idx_test = load_multilayer_data(dataset, use_feats)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data



def load_dhg_data(dataset_str, use_feats):
    """
    根据输入的 dataset_str 字符串，选择对应的数据集类加载数据，
    并返回邻接矩阵、特征、标签以及训练、验证、测试集的索引。
    
    参数:
    - dataset_str (str): 数据集的名称（如 'CoauthorshipCora'、'CoauthorshipDBLP' 等）
    - use_feats (bool): 是否使用特征

    返回:
    - adj: 邻接矩阵 csr_array
    - features: 特征矩阵 tensor
    - labels: 标签 tensor
    - idx_train: 训练集索引 0,1,2,3,..
    - idx_val: 验证集索引 range(140, 640)
    - idx_test: 测试集索引 (1708, ..)
    - adj_train_norm :tensor
    len()=7
    """

    # 根据 dataset_str 加载对应的数据集
    if dataset_str == "CoauthorshipCora":
        hg_data = CoauthorshipCora()
    elif dataset_str == "CoauthorshipDBLP":
        hg_data = CoauthorshipDBLP()
    elif dataset_str == "CocitationCora":
        hg_data = CocitationCora()
    elif dataset_str == "CocitationCiteseer":
        hg_data = CocitationCiteseer()
    elif dataset_str == "CocitationPubmed":
        hg_data = CocitationPubmed()
    else:
        raise ValueError(f"未知的数据集名称：{dataset_str}")

    G = Hypergraph(hg_data["num_vertices"], hg_data["edge_list"])
    # 加载数据

    tensor_adj = G.A.to_dense()  # 邻接矩阵
    dense_array = tensor_adj.numpy()
    sparse_adj = csr_matrix(dense_array)

    features = hg_data["features"] if use_feats else None  # 特征矩阵
    labels = hg_data["labels"]  # 标签

    bool_idx_train, bool_idx_val, bool_idx_test = hg_data["train_mask"], hg_data["val_mask"], hg_data["test_mask"]  # 训练/验证/测试集索引
    idx_train = torch.nonzero(bool_idx_train).squeeze().tolist()
    idx_val = torch.nonzero(bool_idx_val).squeeze().tolist()
    idx_test = torch.nonzero(bool_idx_test).squeeze().tolist()
    return sparse_adj, features, labels, idx_train, idx_val, idx_test


def load_multilayer_data(dataset_str, use_feats):
    if dataset_str == 'CoauthorshipCombinate':
        mhg_data = CoauthorshipCombinate()

        hg1 = Hypergraph(CoauthorshipCora()['num_vertices'],
                         CoauthorshipCora()['edge_list'])
        hg2 = Hypergraph(CoauthorshipDBLP()['num_vertices'],
                         CoauthorshipDBLP()['edge_list'])

        t = MultilayerHypergraph(num_v=mhg_data['num_vertices'],
                                 num_layers=2,
                                 layers_list=[hg1, hg2])
        mhg = t.construct_multi_layer_hypergraph()

    elif dataset_str == 'CocitationCombinate':
        mhg_data = CocitationCombinate()

        hg1 = Hypergraph(CocitationCora()['num_vertices'],
                         CocitationCora()['edge_list'])
        hg2 = Hypergraph(CocitationCiteseer()['num_vertices'],
                         CocitationCiteseer()['edge_list'])
        hg3 = Hypergraph(CocitationPubmed()['num_vertices'],
                         CocitationPubmed()['edge_list'])
        t = MultilayerHypergraph(num_v=mhg_data['num_vertices'],
                                 num_layers=3,
                                 layers_list=[hg1, hg2, hg3])
        mhg = t.construct_multi_layer_hypergraph()
    else:
        raise ValueError(f"未知的数据集名称：{dataset_str}")
    tensor_adj = mhg.A.to_dense()
    dense_array = tensor_adj.numpy()
    sparse_adj = csr_matrix(dense_array)

    features = mhg_data['features']
    labels = mhg_data['labels'] if use_feats else None

    bool_idx_train, bool_idx_val, bool_idx_test = mhg_data["train_mask"], \
                                                  mhg_data["val_mask"],\
                                                  mhg_data["test_mask"]  # 训练/验证/测试集索引
    idx_train = torch.nonzero(bool_idx_train).squeeze().tolist()
    idx_val = torch.nonzero(bool_idx_val).squeeze().tolist()
    idx_test = torch.nonzero(bool_idx_test).squeeze().tolist()
    return sparse_adj, features, labels, idx_train, idx_val, idx_test

def load_single_bio_data(dataset_str, use_feats):
    if dataset_str == "DrugBank":
        hg_data = DrugBank()
    elif dataset_str == "DisGenet":
        hg_data = DisGenet()
    else:
        raise ValueError(f"未知的数据集名称：{dataset_str}")
    G = Hypergraph(hg_data["num_vertices"], hg_data["edge_list"])

    tensor_adj = G.A.to_dense()
    dense_array = tensor_adj.numpy()
    sparse_adj = csr_matrix(dense_array)
    # 如果不使用特征，则创建一个单位特征矩阵
    features = np.ones((sparse_adj.shape[0], 1))
    return sparse_adj, features

def load_bio_data(dataset_str, use_feats):
    if dataset_str == 'biodata':
        # 报错了，没有特征无法用sum计算
        hg1 = Hypergraph(DrugBank()['num_vertices'],
                         DrugBank()['edge_list'])
        hg2 = Hypergraph(DisGenet()['num_vertices'],
                         DisGenet()['edge_list'])

        t = MultilayerHypergraph(num_v=DrugBank()['num_vertices']+DisGenet()['num_vertices'],
                                 num_layers=2,
                                 layers_list=[hg1, hg2])
        mhg = t.construct_multi_layer_hypergraph()

    else:
        raise ValueError(f"未知的数据集名称：{dataset_str}")
    tensor_adj = mhg.A.to_dense()
    dense_array = tensor_adj.numpy()
    sparse_adj = csr_matrix(dense_array)
    # 如果不使用特征，则创建一个单位特征矩阵
    features = np.ones((sparse_adj.shape[0], 1))
    return sparse_adj, features

