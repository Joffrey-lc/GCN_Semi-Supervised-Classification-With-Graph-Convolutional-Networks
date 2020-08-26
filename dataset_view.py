from MyNode import *
import pickle

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



def read_data(path):
    out = pickle.load(open(path, "rb"), encoding="latin1")
    out = out.toarray() if hasattr(out, "toarray") else out
    return out


# adj_dict = read_data("node_classify/cora/cora/raw/ind.cora.graph")
"""根据邻接表创建邻接矩阵"""
# edge_index = []
# num_nodes = len(adj_dict)
# adj = torch.zeros(num_nodes, num_nodes)
# for i, j in adj_dict.items():
#     print(i, j)
#     adj[i, j] = 1
# print(adj.size())
# print(adj[2693:2696, 2693:2696])


# """根据邻接表创建邻接矩阵"""
# num_nodes = len(adj_dict)
# adj = torch.zeros(num_nodes, num_nodes)
# for i, j in adj_dict.items():
#     # print(i, j)
#     degree_test = len(j)
#     adj[i, j] = 1
# # adj = adj + torch.eye(num_nodes, num_nodes)
# degree = torch.zeros_like(adj)
# for i in range(num_nodes):
#     degree_num = torch.sum(adj[i, :])
#     degree[i, :] = degree_num
# d_hat = torch.pow(degree, -0.5)
# print(d_hat[0:8])

