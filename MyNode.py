import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch.nn.init as init
import numpy as np
import pickle
import torch.optim as optim
import matplotlib.pyplot as plt


def my_mul(m1, m2, m3):  # 自定义了m1*m2*m3, 矩阵的对应元素相乘，
    mm1 = torch.mul(m1, m2)
    return torch.mul(mm1, m3)


def get_data(folder="node_classify", data_name="cora"):  # 读数据
    dataset = Planetoid(root=folder, name=data_name)
    return dataset


def get_adjacency(path):
    # 得到邻接表，即读取文件ind.cora.graph
    adj_dict = pickle.load(open(path, "rb"), encoding="latin1")
    adj_dict = adj_dict.toarray() if hasattr(adj_dict, "toarray") else adj_dict
    # 根据邻接表创建邻接矩阵adjacency = [2078, 2078]
    num_nodes = len(adj_dict)
    adjacency = torch.zeros(num_nodes, num_nodes)
    for i, j in adj_dict.items():
        # print(i, j)
        adjacency[i, j] = 1
    # 完成归一化 D^(-0.5) * A * D^(-0.5)
    adjacency = adjacency + torch.eye(num_nodes, num_nodes)
    degree = torch.zeros_like(adjacency)
    for i in range(num_nodes):
        degree_num = torch.sum(adjacency[i, :])
        degree[i, :] = degree_num
    d_hat = torch.pow(degree, -0.5)  # D^(-0.5)
    adjacency = my_mul(d_hat, adjacency, d_hat)
    return adjacency


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.zeros(self.input_dim, self.output_dim, requires_grad=True))
        self.weight = init.kaiming_uniform_(self.weight)  # 凯明初始化

    def forward(self, adjacency, input_feature):
        # f = A*H*W
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        return output


class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))  # 两层网络 第一层输出作为第二层的 H
        out = self.gcn2(adjacency, h)
        return out


def plot_loss_with_acc(loss_history, val_acc_history):  # 画图
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


def test(mask, model, tensor_x, tensor_y, adj):
    model.eval()
    with torch.no_grad():
        logits = model(adj, tensor_x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()
    # return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy()
    return accuarcy


def main():
    # 查看数据集
    cora_dataset = get_data()
    print("=========无向非加权图=========")
    print("Cora数据集类别数目:", cora_dataset.num_classes)
    print("Cora数据集节点数目:", cora_dataset.data.num_nodes)
    print("Cora数据集边的数目:", cora_dataset.data.num_edges)
    print("Cora数据集每个节点特征数目:", cora_dataset.data.num_features)
    print("Cora数据集训练集节点数目:", cora_dataset.data.train_mask.numpy().sum())
    print("Cora数据集验证集节点数目:", cora_dataset.data.val_mask.numpy().sum())
    print("Cora数据集测试集节点数目:", cora_dataset.data.test_mask.numpy().sum())
    print("============================")
    # 训练数据

    learning_rate = 0.1
    weight_decay = 5e-4
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_path = "node_classify/cora/raw/ind.cora.graph"
    adj = get_adjacency(graph_path).to(device)
    data = cora_dataset.data.to(device)
    # 模型定义：Model, Loss, Optimizer
    # 网络的输入维度等于数据集特征的个数，输出维度等于数据集类别数
    model = GcnNet(input_dim=cora_dataset.num_features, hidden_dim=16, output_dim=cora_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    x = data.x / data.x.sum(dim=1, keepdims=True)  # 归一化数据，使得每一行和为1
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = data.y[data.train_mask]  # 训练标签
    for epoch in range(epochs):
        logits = model(adj, x)  # 前向传播
        train_mask_logits = logits[data.train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc = test(data.train_mask, model, x, data.y, adj)  # 计算当前模型训练集上的准确率
        val_acc = test(data.val_mask, model, x, data.y, adj)  # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    test_acc = test(data.test_mask, model, x, data.y, adj)  # 测试集正确率
    print("Test accuarcy: ", test_acc.item())
    plot_loss_with_acc(loss_history, val_acc_history)


if __name__ == '__main__':
    main()
