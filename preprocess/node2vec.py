import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec
import torch_geometric
import networkx as nx
import pickle
import tqdm
from networkx.convert_matrix import from_numpy_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 构建一个简单的 NetworkX 图
data_file = '/root/lizhishuai/sim_3ddata/dataset/'
G = pickle.load(open(data_file + "longhua_1.8k.pkl", "rb"))

for i in G.nodes():
    # G.nodes[i].pop("xy", None)
    # if isinstance(G.nodes[i]['osmid_original'], str):
    #     print(G.nodes[i]['osmid_original'])
    #     G.nodes[i]['osmid_original'] = int(G.nodes[i]['osmid_original'].split(",")[0][1:])
    #     print(G.nodes[i]['osmid_original'])
    for attr_name in list(G.nodes[i].keys()):
        del G.nodes[i][attr_name]

for u, v, attrs in G.edges(data=True):
    attrs.clear()
G.graph.clear()
data = torch_geometric.utils.from_networkx(nx.DiGraph(G))
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                    context_size=10, walks_per_node=10,
                    num_negative_samples=1, p=1, q=1, sparse=True).to(device)

num_workers = 4
loader = model.loader(batch_size=128, shuffle=True,
                        num_workers=num_workers)
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001)

def train():
    model.train()
    for _ in range(1500):
        total_loss = 0
        for ix, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss / (ix+1))
    return total_loss / len(loader)



# 获取节点的向量表示
train()
node_emb = model.embedding.weight.cpu().detach().numpy()
# print(node_emb)
pickle.dump(node_emb, open(data_file + "node_embedding_beijing_3.8k.pkl", "wb"))

# G_temp = pickle.load(open("G_real_sim.pkl", "rb"))

# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(node_emb)

# nodes = sorted(G_temp.nodes())
# # xy_dict = nx.get_node_attributes(G_temp, 'xy')
# # xy_values = [xy_dict[node] for node in nodes]

# x_dict = nx.get_node_attributes(G_temp, 'x')
# y_dict = nx.get_node_attributes(G_temp, 'y')
# xy_dict = {key: (x_dict[key], y_dict[key]) for key in x_dict}
# xy_values = [xy_dict[node] for node in nodes]

# num_clusters = 6
# kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(list(xy_values))
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans.labels_, cmap='Set1', alpha=0.7)  # part 1， plot embedding
# plt.savefig("torch_sim_map_cluster_embeddings.png")
# plt.close()

# first_list = [sublist[0] for sublist in xy_values]
# last_list = [sublist[-1] for sublist in xy_values]
# plt.scatter(first_list, last_list, c=kmeans.labels_, cmap='Set1', alpha=0.7)  # part 2， plot map with k-means
# # plt.scatter(first_list, last_list, c=node_colours, cmap='Set1', alpha=0.7)  # part 2， plot map with k-means
# plt.savefig("torch_sim_map_kmeans_embeddings.png")
# plt.close()