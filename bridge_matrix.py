from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse import csgraph
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy import linalg
from scipy.sparse import csr_matrix
import math
from tqdm import tqdm
from numba import jit
import networkx as nx
import pickle as pkl
import sys
sys.setrecursionlimit(10**5)
from scipy.optimize import minimize

from utils_bridges import *
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from GCN import *
from utils import *
import time
from nparray2dict import *
from utils_homophily import *


import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

dataset_name = "cora"

class SchattenNormMinimizer:
    def __init__(self, p=1.5, max_iter=100, tol=1e-4):
        self.p = p
        self.max_iter = max_iter
        self.tol = tol

    def schatten_norm(self, A):
        singular_values = np.linalg.svd(A, compute_uv=False)
        return np.sum(singular_values**self.p)**(1/self.p)

    def generate_random_initial_guess(self, n):
        random_matrix = np.random.rand(n, n)
        return csr_matrix(random_matrix)

    def minimize(self, adjacency_matrix):
        n = adjacency_matrix.shape[0]

        def objective_function(A):
            return self.schatten_norm(A)

        initial_guess = self.generate_random_initial_guess(n)

        def symmetric_constraint(X):
            A = X.reshape((n, n))
            return np.sum(np.abs(A - A.T))

        constraints = [{'type': 'eq', 'fun': symmetric_constraint}]

        result = minimize(objective_function, initial_guess.toarray().ravel(), method='trust-constr', constraints=constraints)

        optimal_adjacency_matrix = np.round(result.x.reshape((n, n))).astype(int)

        edges_count_input = np.count_nonzero(adjacency_matrix) // 2
        edges_count_output = np.count_nonzero(optimal_adjacency_matrix) // 2

        # Calculate the matrix rank
        rank_input = np.linalg.matrix_rank(adjacency_matrix.toarray())
        rank_output = np.linalg.matrix_rank(optimal_adjacency_matrix)

        return edges_count_input, edges_count_output, rank_input, rank_output, optimal_adjacency_matrix


class Tarjan_Bridges:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        self.visited = [False] * self.num_nodes
        self.ids = [-1] * self.num_nodes
        self.low = [-1] * self.num_nodes
        self.time = 0
        self.bridges = []

    def dfs(self, at, parent):
        self.visited[at] = True
        self.ids[at] = self.time
        self.low[at] = self.time
        self.time += 1

        for to in range(self.num_nodes):
            if self.adj_matrix[at, to] == 0 or to == parent:
                continue

            if not self.visited[to]:
                self.dfs(to, at)
                self.low[at] = min(self.low[at], self.low[to])

                if self.ids[at] < self.low[to]:
                    self.bridges.append((at, to))

            else:
                self.low[at] = min(self.low[at], self.ids[to])

    def find_bridges(self):
        for i in range(self.num_nodes):
            if not self.visited[i]:
                self.dfs(i, -1)

        return self.bridges

cur_path = os.path.abspath('.')
for path in ["cora","citeseer","pubmed"]:
    if not os.path.exists(os.path.join(cur_path, 'GF_Attack_logs', path)):
            os.makedirs(os.path.join(cur_path, 'GF_Attack_logs', path))

parser = ArgumentParser("rdlink_gcn",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("-dataset", required=True, help='dataset string.') # 'citeseer', 'cora', 'pubmed', 
args = parser.parse_args()

_A_obs, _X_obs, _z_obs, y_train, y_val, y_test, train_mask, val_mask, test_mask = calculate.load_data(dataset_str=dataset_name)

adj = _A_obs
train_index = np.where(train_mask)[0]
val_index = np.where(val_mask)[0]
combined_index = np.concatenate((train_index, val_index))
adj_train = adj[train_index, :][:, train_index]


num_train = adj_train.shape[0]
dataset = args.dataset

perturb_save_logs = os.path.join(cur_path, 'EPD_logs/' + dataset + '/Edge_Perturbation_Detector_bestone.txt')
edge_count = adj_train.nnz

A_I = adj_train + sp.eye(adj_train.shape[0])
A_I[A_I > 1] = 1
rowsum = A_I.sum(1)
degree_mat = sp.diags(rowsum)
eig_vals, eig_vec = linalg.eigh(A_I.todense(), degree_mat.todense())
if _X_obs is not None:
    X_mean = np.sum(_X_obs, axis = 1)
else:
    pass

pbar = tqdm(np.arange(adj_train.shape[0], dtype=np.int))
num_nodes = adj.shape[0]
print(adj.nnz)
print(adj_train.nnz)
exclude_nodes = range(num_nodes - 1000, num_nodes)
ori_components = cpt_component(adj)

tb = Tarjan_Bridges(adj)
bridges = tb.find_bridges()

filtered_bridge_edges = [edge for edge in bridges if edge[0] not in exclude_nodes and edge[1] not in exclude_nodes]
bridges = filtered_bridge_edges
num_to_remove = len(bridges)
print(num_to_remove)
edges_to_remove = bridges

tmp_adj = np.array(adj.todense())
np.save(dataset_name +"_bridges.npy", edges_to_remove)


minimal = SchattenNormMinimizer(p=1)
ori_edges, aft_edges, ori_rk, aft_rk = minimal.minimize(tmp_adj)
print("Inputs edges:", ori_edges)
print("Outputs edges:", aft_edges)
print("Input Matrix Rank:", ori_rk)
print("Output Matrix Rank:", aft_rk)
edges_to_remove = np.load(dataset_name +"_bridges.npy", allow_pickle=True)
edges_to_remove = edges_to_remove.tolist()

for edge in edges_to_remove:
    node1, node2 = edge[0], edge[1]
    tmp_adj[node1, node2] = 0
    tmp_adj[node2, node1] = 0
tmp_components = cpt_component(tmp_adj)

print(edges_to_remove)
print(type(edges_to_remove))

adj_full_numpy = tmp_adj


adj_full_numpy = atk(dataset_name, tmp_adj, 500, 0,(0,0))
graph_attributes(nx.from_numpy_array(adj_full_numpy))

adj_forcas = adj_full_numpy + np.eye(adj_full_numpy.shape[0])
cas(adj.toarray(),adj_forcas, dataset_name)

np.save("adj_atk",adj_full_numpy)