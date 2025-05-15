import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc, precision_score, confusion_matrix


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks with two adjacency matrices.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        # 使用动态方法获取流形
        self.manifold = getattr(manifolds, self.manifold_name)()

        # 如果使用Hyperboloid流形，增加一个维度
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj_train, adj_line_graph=None):
        """
        使用两个邻接矩阵 (adj_train: 超图邻接矩阵, adj_line_graph: 线图邻接矩阵) 进行编码。
        """
        # 处理Hyperboloid流形
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1) # 2708, 1434

        # 使用超图邻接矩阵进行编码
        h_train = self.encoder.encode(x, adj_train)
        
        # 使用线图邻接矩阵进行编码
        # h_line_graph = self.encoder.encode(x, adj_line_graph)

        # 将两个嵌入合并成一个
        output = h_train
        return output
    
    #定义一个抽象的compute_metrics方法，子类需要实现这个方法来计算特定任务的度量。
    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Node classification model supporting two adjacency matrices (hypergraph and line graph).
    """
    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)

        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h_train, adj_train, adj_line_graph, idx):
        """
        解码节点嵌入。分别解码超图和线图的嵌入，并结合结果。
        """
        # 计算基于超图邻接矩阵的输出
        output_train = self.decoder.decode(h_train, adj_train)
        
        # 计算基于线图邻接矩阵的输出
        # output_line_graph = self.decoder.decode(h_line_graph, adj_line_graph)

        # 融合两个输出（简单求和或者其他策略）
        # output = output_train + output_line_graph  # 或者你可以尝试其他融合方式
        output = output_train
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        h_train = embeddings  # 获取超图和线图的嵌入
        output = self.decode(h_train, data['adj_train_norm'], data['adj_line_norm'], idx)

        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        labels_cpu = data['labels'][idx].cpu()
        output_cpu = output.max(1)[1].cpu()

        precision = precision_score(labels_cpu, output_cpu, average=self.f1_average)
        recall = recall_score(labels_cpu, output_cpu, average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        return metrics
    
    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

class LPModel(BaseModel):
    """
    Link prediction model supporting two adjacency matrices (hypergraph and line graph).
    """
    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges

    def decode(self, h_train, h_line_graph, idx):
        """
        解码节点嵌入以进行链接预测，考虑超图和线图的嵌入。
        """
        emb_in_train = h_train[idx[:, 0], :]
        emb_out_train = h_train[idx[:, 1], :]
        sqdist_train = self.manifold.sqdist(emb_in_train, emb_out_train, self.c)
        probs_train = self.dc.forward(sqdist_train)

        emb_in_line_graph = h_line_graph[idx[:, 0], :]
        emb_out_line_graph = h_line_graph[idx[:, 1], :]
        sqdist_line_graph = self.manifold.sqdist(emb_in_line_graph, emb_out_line_graph, self.c)
        probs_line_graph = self.dc.forward(sqdist_line_graph)

        # 融合两个模型的预测（简单求和或其他方法）
        probs = probs_train + probs_line_graph  # 或者其他融合方式
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']

        h_train, h_line_graph = embeddings  # 获取超图和线图的嵌入
        pos_scores = self.decode(h_train, h_line_graph, data[f'{split}_edges'])
        neg_scores = self.decode(h_train, h_line_graph, edges_false)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()

        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics
    
    #init_metric_dict 方法用于初始化度量字典。
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}
    #has_improved 方法用于比较两个度量字典，并判断模型性能是否有所改进。
    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

