import time
import pickle as pkl
import torch
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from data import MultilayerHypergraph
from data import CoauthorshipCombinate, CocitationCombinate

from dhg.models import HGNN, HGNNP, HyperGCN, UniGCN, UniGAT
from dhg import Hypergraph
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.data import (CoauthorshipCora,
                      CoauthorshipDBLP,
                      CocitationCiteseer,
                      CocitationCora,
                      CocitationPubmed)


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(
        f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    set_seed(2021)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = Evaluator(
        ["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    # data = CoauthorshipCombinate()
    data = CocitationCombinate()

    X, lbl = data['features'], data['labels']
    # g_cora = Hypergraph(CoauthorshipCora()['num_vertices'],CoauthorshipCora()['edge_list'])
    # g_dblp = Hypergraph(CoauthorshipDBLP()['num_vertices'],CoauthorshipDBLP()['edge_list'])
    
    g_cora = Hypergraph(
        CocitationCora()['num_vertices'],
        CocitationCora()['edge_list'])
    g_citeseer = Hypergraph(
        CocitationCiteseer()['num_vertices'],
        CocitationCiteseer()['edge_list'])
    g_pubmed = Hypergraph(
        CocitationPubmed()['num_vertices'],
        CocitationPubmed()['edge_list'])
    

    # _G = MultilayerHypergraph(num_v=data["num_vertices"],
    #                          num_layers=2,
    #                          layers_list=[g_cora, g_dblp])

    _G = MultilayerHypergraph(num_v=data["num_vertices"],
                              num_layers=3,
                              layers_list=[g_cora, g_citeseer, g_pubmed])
    G = _G.construct_multi_layer_hypergraph()
    print(G)

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
    net = HyperGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.to(device), lbl.to(device)
    G = G.to(device)
    net = net.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(net, X, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, X, G, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
