import scipy.sparse as sp
from scipy.sparse import csgraph
import numpy as np
from scipy import linalg
from tqdm import tqdm
import tensorflow as tf
from utils_bridges import *
import os
import argparse
#from config import *
from utils import *
from nparray2dict import *
from utils_homophily import *



parser = argparse.ArgumentParser(description='INPUT')
parser.add_argument('-heteE', type=int, help='number of perturbed hete edges')
parser.add_argument('-homoE', type=int, help='number of perturbed homo edges')
parser.add_argument('-dataset', type=str, default='cora' ,help='the used dataset')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs')
parser.add_argument('-rn', type=int, default=1, help='number of edges to be removed per epoch')
parser.add_argument('-type', type=str, default='atk', help='the cases of augmentation or attack')
args = parser.parse_args()

dtype = tf.float32

_A_obs, _X_obs, _z_obs, y_train, y_val, y_test, train_mask, val_mask, test_mask = calculate.load_data(dataset_str= args.dataset)

def nonzero_diagonal_idx(matrix):
    diagonal_indices = np.where(np.diag(matrix) != 0)[0]
    return len(diagonal_indices)


#compt_metric(args.dataset)
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

n_components, labels = csgraph.connected_components(adj.toarray())
for i in range(n_components):
    component = np.where(labels == i)[0]
print(n_components)


if args.type == 'aug':
    homo_mt = np.load(args.dataset+"_homo_mt.npy", allow_pickle=True)
    
    init_temperature = 2.0
    temperature_decay = 0.99
    adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
    all_labels = y_train + y_test+y_val
    single_label = np.argmax(all_labels,axis=-1)
    nodesize = features.shape[0]
    
    #Fetch adj_train for processing
    train_index = np.where(train_mask)[0]
    val_index = np.where(val_mask)[0]
    com_index = np.concatenate((train_index, val_index))
    adj_train = adj[train_index,:][:,train_index]
    print("Shape of adj_train:", adj_train.shape)
    adj_tmp = np.zeros((adj.shape[0],adj.shape[0]))
    # Some preprocessing
    features = preprocess_features(features)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    tuple_adj = sparse_to_tuple(adj_train.tocoo())
    features_tensor = tf.convert_to_tensor(features,dtype=dtype)
    adj_tensor = tf.SparseTensor(*tuple_adj)
    
    model = AttentionModel(input_dim=features.shape[1], output_dim=y_train.shape[1])
    model.set_fea_adj(np.array(range(adj_train.shape[0])), features_tensor, adj_tensor)
    
    adj_np = model.SpTensor2Numpy("adj")
    orig_edge = np.count_nonzero(adj_np)
    print("#"*30)
    print("The Information of Adj at Start:")
    print("Adj Shape:", adj_np.shape)
    print("Adj Nonzero Num:", orig_edge)
    print("#"*30)
    
    min_rank = 10000
    target_adj_train = None

    for e in range(args.epochs):
        with tf.GradientTape() as tape:
            temperature = max(0.05, init_temperature * pow(temperature_decay, e))
            model.call(temperature, training=True)
            loss = model.nuclear_loss(4)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        edges_volumn = tf.reduce_sum(model.maskes[0])
        print("-"*20, "Epoch:", e, "-"*20)
        avg = anlysis(model.maskes[0].numpy())
        print("Edge Vol:", edges_volumn.numpy(), ", Avg Mask:", avg, ", Nuclear Loss:", loss)
        mask = np.array(model.maskes[0], dtype='float32')
        mask = tf.convert_to_tensor(mask)
        indices, values = model.update(new_mask = tf.squeeze(mask))
        model.filter_adj(indices, values, args.rn, homo_mt)
        adj_np_ed = model.SpTensor2Numpy("sparse_adj")
        print("-"*51)
    target_adj_train = adj_np_ed
    adj_tmp[:target_adj_train.shape[0], :target_adj_train.shape[0]] = target_adj_train
    adj_tmp[target_adj_train.shape[0]:, target_adj_train.shape[0]:] = adj.toarray()[target_adj_train.shape[0]:, target_adj_train.shape[0]:]                   
    print("Totally,", (orig_edge-np.count_nonzero(target_adj_train))/2, "edges are removed")
    print("Save the Corresponding adj_train")
    adj_tmp = aug(args.dataset, adj.toarray(), args.heteE, args.homoE)
    np.save(args.dataset+"_adj_"+args.type, adj_tmp)
    adj_forcas = adj_tmp + np.eye(adj_tmp.shape[0])
    cas(adj.toarray(),adj_forcas, args.dataset)
    np.save(args.dataset+"_adj_"+args.type, adj_forcas)

    
elif args.type == 'atk':
    num_nodes = adj.shape[0]
    exclude_nodes = range(num_nodes - 1000, num_nodes)
    tb = Tarjan_Bridges(adj)
    bridges = tb.find_bridges()

    filtered_bridge_edges = [edge for edge in bridges if edge[0] not in exclude_nodes and edge[1] not in exclude_nodes]
    bridges = filtered_bridge_edges
    num_to_remove = len(bridges)
    edges_to_remove = bridges
    adj_full_numpy = atk(args.dataset, adj.toarray(), args.heteE, args.homoE, edges_to_remove)
    adj_forcas = adj_full_numpy + np.eye(adj_full_numpy.shape[0])
    cas(adj.toarray(), adj_forcas, args.dataset)
    np.save(args.dataset+"_adj_"+args.type, adj_full_numpy)
else:
    print("Invalid Input")






