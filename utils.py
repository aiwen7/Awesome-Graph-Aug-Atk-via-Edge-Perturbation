# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
import networkx as nx
import pickle as pkl
import sys
#from config import *

SVD_PI = True
eps = 1e-7
class AttentionModel(keras.Model):
    def __init__(self, input_dim, output_dim):
        super(AttentionModel, self).__init__(self, input_dim, output_dim)
        self.nb_layer = tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal())
        self.selflayer = tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal())
        self.attentions = []
        self.edge_weights = []
        self.support = None
        for i in range(2):
            self.attentions.append([])
            self.attentions[i].append(tf.keras.layers.Dense(1, activation=lambda x:x, kernel_initializer=tf.keras.initializers.HeNormal()))
    
    def get_attention(self, input1, input2 ,training=False):
        nn = self.attentions[0]
        dp = 0.5
        input1 = self.nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = self.selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)
        input10 = tf.concat([input1, input2], axis=1)
        input_ = [input10]
        for layer in nn:
            input_.append(layer(input_[-1]))
            if training:
                input_[-1] = tf.nn.dropout(input_[-1], dp)
        weight10 = input_[-1]
        return weight10
    
    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        gamma = -0.0
        zeta = 1.01
        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)
        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(stretched_values, clip_value_max=1.0, clip_value_min=0.0)
        return cliped
        
    def set_fea_adj(self,nodes,fea,adj):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat = adj
        self.row = adj.indices[:,0]
        self.col = adj.indices[:,1]
        self.support = None
        self.sparse_adj = self.adj_mat
    
    def SpTensor2Numpy(self, flag):
        #print("Original Tpye", type(self.adj_mat))
        if flag == 'adj':
            tmp = self.adj_mat
        elif flag == 'support':
            tmp = self.support
        elif flag == 'sparse_adj':
            tmp = self.sparse_adj
        else:
            print("Invalid Input")
            return
        tmp = tf.sparse.to_dense(tmp).numpy() #tensorflow.python.framework.ops.EagerTensor -> numpy
        return tmp
    
    def nuclear_loss(self, k_svd):
        nuclear_loss = tf.zeros([],dtype='float32')
        values = []
        for mask in self.maskes:
            mask = tf.squeeze(mask)
            support = tf.SparseTensor(indices=self.adj_mat.indices, values=mask, dense_shape=self.adj_mat.dense_shape)
            support_dense = tf.sparse.to_dense(support)  #adj矩阵先转为稠密矩阵(确保一些矩阵运算可以执行)
            support_trans = tf.transpose(support_dense)  #adj的转置矩阵adj*
            AA = tf.matmul(support_trans, support_dense) #adj*×adj记为AA
            #PI近似方法从随机初始化的向量出发来计算AA的最大特征值与主特征向量，adj的最大奇异值即最大特征值的平方根
            if SVD_PI:
                row_ind = self.adj_mat.indices[:, 0]
                col_ind = self.adj_mat.indices[:, 1]
                support_csc = csc_matrix((mask.numpy(), (row_ind.numpy(), col_ind.numpy())))
                k = k_svd
                u, s, vh = svds(support_csc, k=k) #计算获取稀疏矩阵的最大(小)k个奇异值与对应的奇异向量
                u = tf.stop_gradient(u)   #在Pytorch中等价于detach()函数
                s = tf.stop_gradient(s)
                vh = tf.stop_gradient(vh) #右奇异向量(每行都是奇异向量)
                for i in range(k): #利用幂迭代来计算top k奇异值
                    vi = tf.expand_dims(tf.gather(vh, i), -1) #gather为切片函数，expand_dims为从末尾维度进行维度拓展
                    for ite in range(1):
                        vi = tf.matmul(AA, vi)
                        vi_norm = tf.linalg.norm(vi)
                        vi = vi / vi_norm
                    vmv = tf.matmul(tf.transpose(vi), tf.matmul(AA, vi))
                    vv = tf.matmul(tf.transpose(vi), vi)
                    t_vi = tf.math.sqrt(tf.abs(vmv / vv)) #获取top K(i-th)特征值
                    values.append(t_vi)
                    if k > 1:
                        AA_minus = tf.matmul(AA, tf.matmul(vi, tf.transpose(vi)))
                        AA = AA - AA_minus
            else:
                trace = tf.linalg.trace(AA)
                values.append(tf.reduce_sum(trace))
            nuclear_loss = tf.add_n(values) #逐元素加法
        return nuclear_loss
        
    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0
        self.edge_maskes = []
        self.maskes = []
        x = self.features
        layer_index = 0
        f1_features = tf.gather(x, self.row)
        f2_features = tf.gather(x, self.col)
        weight = self.get_attention(f1_features, f2_features, training=training)
        mask = self.hard_concrete_sample(weight, temperature, training)
        mask_sum = tf.reduce_sum(mask)
        self.edge_weights.append(weight)
        self.maskes.append(mask)
        
    def update(self, new_mask):
        adj = tf.SparseTensor(indices=self.adj_mat.indices, values=new_mask, dense_shape=self.adj_mat.shape)
        # norm
        adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype='float32'))
        row = adj.indices[:, 0]
        col = adj.indices[:, 1]
        rowsum = tf.sparse.reduce_sum(adj, axis=-1)
        d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
        d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
        row_inv_sqrt = tf.gather(d_inv_sqrt,row)
        col_inv_sqrt = tf.gather(d_inv_sqrt,col)
        values = tf.multiply(adj.values,row_inv_sqrt)
        values = tf.multiply(values,col_inv_sqrt)
        self.support = tf.SparseTensor(indices=adj.indices, values=values, dense_shape=adj.shape)
        return self.support.indices, self.support.values
        
    def filter_adj(self, indices, values, top_k_small, homo_mt):
        values = values.numpy()
        smallest_val_ind = np.argpartition(values, top_k_small)[:top_k_small]
        #print("+++++++ TEST +++++++")
        smallest_indices = tf.gather(indices, smallest_val_ind).numpy() #TF Tensor->Numpy
        ### Add the symmetric edges ###
        tmp_indices = []
        for ind in smallest_indices:
            tmp_indices.append([ind[1], ind[0]])
        tmp_indices = np.array(tmp_indices)
        smallest_indices = np.concatenate((smallest_indices, tmp_indices), axis=0)
        Orig_adj_indices = self.sparse_adj.indices.numpy()
        Orig_adj_values = self.sparse_adj.values.numpy()
        #print(Orig_adj_indices)
        #print(Orig_adj_values)
        target_ind = []
        for ind in smallest_indices:
            if ind[0] == ind[1]:
                continue
            for i in range(len(Orig_adj_indices)):
                if (ind == Orig_adj_indices[i]).all():
                    if homo_mt[ind[0]][ind[1]] == 1: #the edge is heterophilic
                        target_ind.append(i)
        print("Remove", len(target_ind), "Edges from adj_mat")
        Orig_adj_indices = np.delete(Orig_adj_indices, target_ind, 0)
        Orig_adj_values = np.delete(Orig_adj_values, target_ind, 0)
        #print(len(Orig_adj_indices))
        #print(len(Orig_adj_values))
        adj_indices = tf.convert_to_tensor(Orig_adj_indices)
        adj_values = tf.convert_to_tensor(Orig_adj_values)
        self.sparse_adj = tf.SparseTensor(indices = adj_indices, values = adj_values, dense_shape = self.adj_mat.shape)
        
    def update1(self, new_mask):
        adj = tf.SparseTensor(indices=self.support.indices, values=new_mask, dense_shape=self.adj_mat.shape)
        # norm
        adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype='float32'))
        row = adj.indices[:, 0]
        col = adj.indices[:, 1]
        rowsum = tf.sparse.reduce_sum(adj, axis=-1)
        d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
        d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
        row_inv_sqrt = tf.gather(d_inv_sqrt,row)
        col_inv_sqrt = tf.gather(d_inv_sqrt,col)
        values = tf.multiply(adj.values,row_inv_sqrt)
        values = tf.multiply(values,col_inv_sqrt)
        self.support = tf.SparseTensor(indices=adj.indices, values=values, dense_shape=adj.shape)
        print(type(self.support.indices.numpy()), type(self.support.values.numpy()))
        
    def update2(self, new_indices, new_mask):
        adj = tf.SparseTensor(indices=new_indices, values=new_mask, dense_shape=self.adj_mat.shape)
        # norm
        adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype='float32'))
        row = adj.indices[:, 0]
        col = adj.indices[:, 1]
        rowsum = tf.sparse.reduce_sum(adj, axis=-1)#+1e-6
        d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
        d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
        row_inv_sqrt = tf.gather(d_inv_sqrt,row)
        col_inv_sqrt = tf.gather(d_inv_sqrt,col)
        values = tf.multiply(adj.values,row_inv_sqrt)
        values = tf.multiply(values,col_inv_sqrt)
        self.support = tf.SparseTensor(indices=adj.indices, values=values, dense_shape=adj.shape)

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


def load_npz_edges(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    dict_of_lists = {}
    with np.load(file_name) as loader:
        loader = dict(loader)
        num_nodes = loader['adj_shape'][0]
        indices = loader['adj_indices']
        indptr = loader['adj_indptr']
        for i in range(num_nodes):
            if len(indices[indptr[i]:indptr[i+1]]) > 0:
                dict_of_lists[i] = indices[indptr[i]:indptr[i+1]].tolist()

    return dict_of_lists


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = adj_.sum(1)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def cal_scores(A, X_mean, eig_vals, eig_vec, filtered_edges, K=2, T=128, lambda_method = "nosum"):
    '''
    Calculate the scores as formulated in paper.

    Parameters
    ----------
    K: int, default: 2
        The order of graph filter K.

    T: int, default: 128
        Selecting the Top-T smallest eigen-values/vectors.

    lambda_method: "sum"/"nosum", default: "nosum"
        Indicates the scores are calculated from which loss as in Equation (8) or Equation (12).
        "nosum" denotes Equation (8), where the loss is derived from Graph Convolutional Networks,
        "sum" denotes Equation (12), where the loss is derived from Sampling-based Graph Embedding Methods.

    Returns
    -------
    Scores for candidate edges.

    '''
    results = []
    A = A + sp.eye(A.shape[0])
    A[A > 1] = 1
    rowsum = A.sum(1).A1
    D_min = rowsum.min()
    abs_V = len(eig_vals)
    tmp_k = T

    return_values = []

    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        eig_vals_res = np.zeros(len(eig_vals))
        eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:] - eig_vals *
                                                                        ( np.square(eig_vec[filtered_edge[0],:]) + np.square(eig_vec[filtered_edge[1],:])))
        eig_vals_res = eig_vals + eig_vals_res

        if lambda_method == "sum":
            if K==1:
                eig_vals_res =np.abs(eig_vals_res / K) * (1/D_min)
            else:
                for itr in range(1,K):
                    eig_vals_res = eig_vals_res + np.power(eig_vals_res, itr+1)
                eig_vals_res = np.abs(eig_vals_res / K) * (1/D_min)
        else:
            eig_vals_res = np.square((eig_vals_res + np.ones(len(eig_vals_res))))
            eig_vals_res = np.power(eig_vals_res, K)

        eig_vals_idx = np.argsort(eig_vals_res)  # from small to large
        eig_vals_k_sum = eig_vals_res[eig_vals_idx[:tmp_k]].sum()
        u_k = eig_vec[:,eig_vals_idx[:tmp_k]]
        u_x_mean = u_k.T.dot(X_mean)
        return_values.append(eig_vals_k_sum * np.square(np.linalg.norm(u_x_mean)))
    num_nodes = A.shape[0]
    return_values_array = np.array(return_values)
    result_matrix = np.zeros((num_nodes, num_nodes))
    for j in range(len(filtered_edges)):
        filtered_edge = filtered_edges[j]
        result_matrix[filtered_edge[0], filtered_edge[1]] = return_values_array[j]
        result_matrix[filtered_edge[1], filtered_edge[0]] = return_values_array[j]
        print(result_matrix.shape)
    print("calculation done\n")

    return np.array(return_values)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features).astype(np.float32)
    try:
        return features.todense() # [coordinates, data, shape], []
    except:
        return features

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def filter_singles(edges, adj):

    degree = np.squeeze(np.array(np.sum(adj,0)))
    existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
    if existing_edges.size > 0:
        edge_degrees = degree[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
    else:
        edge_degrees = degree[np.array(edges)] + 1

    zeros = edge_degrees == 0
    zeros_sum = zeros.sum(1)
    return zeros_sum == 0

def largest_connected_components(adj, n_components=1):

    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def filter_matrix(matrix, adj, N):
    filtered_data = matrix
    
    #filtered_data = matrix.toarray()
    adj_shape = adj.toarray()
    nonzero_values = filtered_data[np.nonzero(filtered_data)]
    sorted_values = np.sort(nonzero_values)
    
    non_zero_indices = np.argwhere(adj_shape != 0)
    np.random.shuffle(non_zero_indices)
    selected_indices = non_zero_indices[:N]
    
    
    a = sorted_values[N]
    b = sorted_values[-((2*N)-1)]
    mask_a = (filtered_data < a) & (filtered_data != 0)
    mask_b = b < filtered_data  
    filtered_matrix_a = np.where(mask_a, 1, 0)
    filtered_matrix_b = np.where(mask_b, 1, 0)
    '''
    nonzero_indices = np.nonzero(filtered_matrix_b)
    for row,col in zip(*nonzero_indices):
        value = filtered_matrix_b[row,col]
        print(f"Value at index ({row}, {col}): {value}")
    '''
    filtered_a = np.zeros_like(adj_shape)
    print(len(adj_shape[0]))
    filtered_b = np.zeros_like(adj_shape)
    filtered_a[:np.array(filtered_matrix_a).shape[0], :np.array(filtered_matrix_a).shape[1]] = filtered_matrix_a
    filtered_b[:np.array(filtered_matrix_b).shape[0], :np.array(filtered_matrix_b).shape[1]] = filtered_matrix_b
    return filtered_a, filtered_b

def load_data(dataset_str):
    """Load data."""
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index


    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    #idx_test = test_idx_range.tolist()
    #idx_train = range(len(y))
    #idx_val = range(len(y), len(y)+500)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally)-500)
    idx_val = range(len(ally)-500, len(ally))
    
    combined_index = np.concatenate((idx_train, idx_val))
    features_combined = features[combined_index, :]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask

def graph_attributes(graph):
    """to compute graph_attributes after perts.

    Args:
        graph (_type_):  graph
    """
    #graph_before = nx.from_dict_of_lists(graph)
    graph_before = graph
    
    distribute_degree = nx.degree(graph_before)
    neighbor_degree = nx.average_neighbor_degree(graph_before)
    clustering_coeffcient = nx.average_clustering(graph_before)
    global_efficiency = nx.global_efficiency(graph_before)
    eigenvector_centrality = nx.eigenvector_centrality(graph_before)
    closeness_centrality = nx.closeness_centrality(graph_before)
    betweenness_centrality = nx.betweenness_centrality(graph_before)
    degree_centrality = nx.degree_centrality(graph_before)
    
    print("#"*30)
    print("EPD Stage 1:")
    print("#"*30)
    print("global_efficiency:", global_efficiency)
    print("closeness_centrality:", sum(closeness_centrality.values()) / float(len(closeness_centrality)))
    print("degree:", sum(dict(distribute_degree).values()) / float(len(dict(distribute_degree))))
    print("neighbor_degree:", sum(neighbor_degree.values()) / float(len(neighbor_degree)))
    print("eigenvector_centrality:", sum(eigenvector_centrality.values()) / float(len(eigenvector_centrality)))
    print("clustering_coeffcient:", clustering_coeffcient)
    print("betweenness_centrality:", sum(betweenness_centrality.values()) / float(len(betweenness_centrality)))
    print("degree_centrality:", sum(degree_centrality.values()) / float(len(degree_centrality)))
    
def anlysis(inputs):
    return np.mean(inputs)