import time
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import CoauthorshipCora, CoauthorshipDBLP, CocitationCora, CocitationPubmed, CocitationCiteseer
from dhg.models import HGNN, HGNNP, HyperGCN, UniGCN, UniGAT
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
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
    # 记录程序开始时间
    start_time = time.time()
    set_seed(2021)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])

    # data = CoauthorshipCora()
    data = CoauthorshipDBLP()
    # data = CocitationCora()
    # data = CocitationCiteseer()
    # data = CocitationPubmed()
    

    # X, lbl = torch.eye(data["num_vertices"]), data["labels"]
    X, lbl = data['features'], data['labels']
    G = Hypergraph(data["num_vertices"], data["edge_list"])
    # print(G)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]
    # print(X)
    # print(lbl)
    # print(data['edge_list'])
    net = HGNN(X.shape[1], 32, data["num_classes"], use_bn=True)
    # net = HGNNP(X.shape[1], 32, data["num_classes"], use_bn=False)
    # net = HyperGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    # net = UniGCN(X.shape[1], 32, data["num_classes"], use_bn=False)
    # net = UniGAT(X.shape[1], 32, data["num_classes"], num_heads=2, use_bn=False)
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

    with torch.no_grad():
        net.eval()
        outs = net(X, G)
        outs, lbls = outs[test_mask], lbl[test_mask]
        embeddings_cpu = outs.cpu().numpy()
        silhouette_scores = sklearn.metrics.silhouette_score(embeddings_cpu, lbls)  # 计算轮廓分数

        # 使用PCA进行降维至3维
        pca = PCA(n_components=3)
        embeddings_pca = pca.fit_transform(embeddings_cpu)

        # 可视化
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2], c=lbls,
                    cmap='plasma')

        plt.title('3D Embeddings Visualization')
        plt.show()

        print(silhouette_scores)

    print(f"final result: epoch: {best_epoch}")
    print(res)

    # 计算并打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal running time: {total_time:.2f} seconds")
    # 如果需要更详细的时间格式
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total running time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")