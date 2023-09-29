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
from utils_bridges import *
import os
import argparse
from utils import *
from nparray2dict import *
from utils_homophily import *



parser = argparse.ArgumentParser(description='INPUT')
parser.add_argument('-heteE', type=int, help='number of perturbed hete edges')
parser.add_argument('-homoE', type=int, help='number of perturbed homo edges')
parser.add_argument('-dataset', type=str, default='cora' ,help='the used dataset')
parser.add_argument('-type', type=str, default='atk', help='the cases of augmentation or attack')
args = parser.parse_args()

_A_obs, _X_obs, _z_obs, y_train, y_val, y_test, train_mask, val_mask, test_mask = calculate.load_data(dataset_str= args.dataset)

def nonzero_diagonal_idx(matrix):
    diagonal_indices = np.where(np.diag(matrix) != 0)[0]
    return len(diagonal_indices)


adj = _A_obs
train_index = np.where(train_mask)[0]
val_index = np.where(val_mask)[0]
combined_index = np.concatenate((train_index, val_index))
adj_train = adj[train_index, :][:, train_index]

di = nonzero_diagonal_idx(adj_train.toarray())

num_train = adj_train.shape[0]
dataset = args.dataset
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
num_nodes = adj_train.shape[0]
scores_matrix = np.zeros((num_nodes, num_nodes))
    
n_components, labels = csgraph.connected_components(adj.toarray())
for i in range(n_components):
    component = np.where(labels == i)[0]
print(n_components)
compt_metric(args.dataset)

if args.type == 'aug':
    adj_full_numpy = aug(args.dataset, adj.toarray(), args.heteE, args.homoE)
elif args.type == 'atk':
    adj_full_numpy = atk(args.dataset, adj.toarray(), args.heteE, args.homoE, (0,0))
else:
    print("Invalid Input")

adj_forcas = adj_full_numpy + np.eye(adj_full_numpy.shape[0])
cas(adj.toarray(), adj_forcas, args.dataset)

np.save(args.dataset+"_adj_"+args.type, adj_full_numpy)



