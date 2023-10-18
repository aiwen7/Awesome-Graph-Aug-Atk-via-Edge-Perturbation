import numpy as np
import pickle as pkl
import scipy.sparse as sp
import collections
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components

np.set_printoptions(threshold=np.inf)
seed = 42
np.random.seed(seed)

'''
def convert(dataset_name, adj_full_numpy):
    output_list = []
    afn = adj_full_numpy
    row_col_tuple = np.where(afn > 0)
    row = row_col_tuple[0]
    col = row_col_tuple[1]
    for i in range(len(row)):
        output_list.append((row[i], col[i]))
    output_dict = collections.defaultdict(list)
    for key, value in output_list:
        output_dict[key].append(value)
    #print(output_dict)
    write_file = open("ind." + dataset_name + ".graph", "wb")
    pkl.dump(output_dict, write_file)
    write_file.close()
'''
def random_shuffle(a, b):
    length = len(a)
    a = np.reshape(a, (1, length))
    b = np.reshape(b, (1, length))
    concat_ab = np.concatenate((a, b), axis=0)
    transposed_ab = np.transpose(concat_ab)
    np.random.shuffle(transposed_ab)
    result = np.transpose(transposed_ab)
    output_a = result[0]
    output_b = result[1]
    return output_a, output_b

def cas(adj_ori, adj_full_numpy, dataset_name):
    output_list = []
    afn = adj_full_numpy
    row_col_tuple = np.where(afn > 0)
    row = row_col_tuple[0]
    col = row_col_tuple[1]
    
    for i in range(len(row)):
        output_list.append((row[i], col[i]))
    output_dict = collections.defaultdict(list)
    for key, value in output_list:
        output_dict[key].append(value)
    #print(output_dict)
    #for i in output_dict:
    #    print(i)
    write_file = open("ind." + dataset_name + ".graph", "wb")
    pkl.dump(output_dict, write_file)
    write_file.close()
    output_list_ori = []
    afn_ori = adj_ori
    row_col_tuple_ori = np.where(afn_ori > 0)
    row_ori = row_col_tuple_ori[0]
    col_ori = row_col_tuple_ori[1]
    for i in range(len(row_ori)):
        output_list_ori.append((row_ori[i], col_ori[i]))
    output_dict_ori = collections.defaultdict(list)
    for key, value in output_list_ori:
        output_dict_ori[key].append(value)
    diff = output_dict_ori.keys() & output_dict
    diff_vals = [(k, output_dict_ori[k], output_dict[k]) for k in diff if output_dict_ori[k] != output_dict[k]]
    #print(diff_vals)
    missing_keys = []
    missing_values = {}
    for key in output_dict_ori:
        if key not in output_dict:
            missing_keys.append(key)
            missing_values[key] = output_dict_ori[key]
    print("Missing keys are:",missing_keys)
    print("Missing values are:",missing_values)

def load_file(file_path):
    with open(file_path, "rb") as f:
        file = pkl.load(f)
    return file
    
def get_neighbor_idx(node_idx, adj_train_numpy):
    return np.where(adj_train_numpy[node_idx, :]>0)[0]
    
def compt_node_metric(flag, node_label, neighbor_idx_list, label_mt_numpy):
    count = 0
    total_neighbor = len(neighbor_idx_list)
    if flag == 1: #计算同质性
        for i in range(total_neighbor):
            if node_label == label_mt_numpy[neighbor_idx_list[i]]:
                count = count + 1
        result = count/total_neighbor
    else:
        for i in range(total_neighbor):
            if node_label != label_mt_numpy[neighbor_idx_list[i]]:
                count = count + 1
        result = -count/total_neighbor
    return result

def graph_homophily(dataset_name, adj_train_numpy): #要求adj必须没有自环
    label_mt_numpy = load_file("./ind." + dataset_name + ".ally")
    num_node = adj_train_numpy.shape[0]
    node_homophily_list = []
    print(num_node, "Nodes are Considered")
    for i in range(num_node):
        node_label = label_mt_numpy[i]
        neigh_idx = np.where(adj_train_numpy[i]>0)[0]
        total_num_neighbor = len(neigh_idx)
        homo_neigh_count = 0
        for j in range(total_num_neighbor):
            if node_label == label_mt_numpy[neigh_idx[j]].all:
                homo_neigh_count = homo_neigh_count + 1
        node_homophily_list.append(homo_neigh_count/total_num_neighbor)
    graph_homophily = np.average(np.array(node_homophily_list))
    return graph_homophily

def compt_metric(dataset_name): #dataset_name = "cora" / "citeseer" / "pubmed"
    if dataset_name == "cora":
        node_num = 2708
    elif dataset_name == "citeseer":
        node_num = 3312 #3327
    elif dataset_name == "pubmed":
        node_num = 19717
    else:
        print("Invalid Dataset")
        return
    label_mt_numpy = load_file("./data/ind." + dataset_name + ".ally")
    length = label_mt_numpy.shape[0] - 500
    print(length, "nodes are considered ...")
    homo_mt = np.zeros((node_num, node_num))
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            
            elif (label_mt_numpy[i] == label_mt_numpy[j]).all():
                homo_mt[i][j] = 1
            else:
                homo_mt[i][j] = -1    
    np.save(dataset_name+"_homo_mt.npy", homo_mt)
    
def compt_homo_2hop(dataset_name, adj_train_numpy):
    if dataset_name == "cora":
        node_num = 2708
    elif dataset_name == "citeseer":
        node_num = 3327
    elif dataset_name == "pubmed":
        node_num = 19717
    else:
        print("Invalid Dataset")
        return
    label_mt_numpy = load_file("./data/ind." + dataset_name + ".ally") 
    length = label_mt_numpy.shape[0]
    print(length, "nodes are considered ...")
    homo_mt_2hop = np.zeros((node_num, node_num))
    for i in range(length):
        neighbor_idx_list =  get_neighbor_idx(i, adj_train_numpy)#第i行，就获取i的邻居
        node_label = label_mt_numpy[i]
        for j in range(len(neighbor_idx_list)):
            flag = 0
            if (node_label == label_mt_numpy[neighbor_idx_list[j]]).all:
                flag = 1
            metric = compt_node_metric(flag, node_label, neighbor_idx_list, label_mt_numpy)
        homo_mt_2hop[i][j] = metric
    return homo_mt_2hop

def preload_idx(datasetname,path):
    homo_row_idx = np.load("/home/liuxin/IED/" + datasetname + "homo_row_idx.npy", allow_pickle=True)
    homo_col_idx = np.load("/home/liuxin/IED/" + datasetname + "homo_col_idx.npy", allow_pickle=True)
    hete_row_idx = np.load("/home/liuxin/IED/" + datasetname + "hete_row_idx.npy", allow_pickle=True)
    hete_col_idx = np.load("/home/liuxin/IED/" + datasetname + "hete_col_idx.npy", allow_pickle=True)
    return homo_row_idx, homo_col_idx, hete_row_idx, hete_col_idx

def atk(datasetname, adj_full_numpy, add_hete_budget, rm_homo_budget, edges_to_remove):
    #Step 1
    #在attack场景下 优先添加异质边来进行攻击
    homo_row_idx, homo_col_idx, hete_row_idx, hete_col_idx = preload_idx(datasetname, "./")
    max_idx = np.max(hete_row_idx) if np.max(hete_row_idx) > np.max(hete_col_idx) else np.max(hete_col_idx)
    print(len(homo_row_idx)+len(hete_row_idx))
    print("*"*30)
    if max_idx <= (adj_full_numpy.shape[0] - 1000):
        print("No Test Nodes will be Affected in Adding hete Edges")
    else:
        print("Invalid Node Index")
        return
    add_hete_count = 0
    for i in range(len(hete_row_idx)):
        #if hete_row_idx[i] <= hete_col_idx[i]:
        #    continue #如果这个位置在对角线或者对角线上面则不考虑
        #else:
        if add_hete_budget == 0:
            break
        if (hete_row_idx[i], hete_col_idx[i]) in edges_to_remove or (hete_col_idx[i], hete_row_idx[i]) in edges_to_remove:
            continue
        if adj_full_numpy[hete_row_idx[i]][hete_col_idx[i]] == 0: #在异质性的边位置无连接
            adj_full_numpy[hete_row_idx[i]][hete_col_idx[i]] = 1  #添加该异质性的边
            adj_full_numpy[hete_col_idx[i]][hete_row_idx[i]] = 1  #添加对称边
            add_hete_count = add_hete_count + 1
            if add_hete_count == add_hete_budget:
                break
    #Step 2
    #减少特定数量的同质性边【可能】会给带来攻击效果
    max_idx = np.max(homo_row_idx) if np.max(homo_row_idx) > np.max(homo_col_idx) else np.max(homo_col_idx)
    if max_idx <= (adj_full_numpy.shape[0] - 1000):
        print("No Test Nodes will be Affected in Removing homo Edges")
    else:
        print("Invalid Node Index")
        return
    rm_homo_count = 0
    for i in range(len(homo_row_idx)):
        #if hete_row_idx[i] <= hete_col_idx[i]:
        #    continue #如果这个位置在对角线或者对角线上面则不考虑
        #else:
        if rm_homo_budget == 0:
            break
        if adj_full_numpy[homo_row_idx[i]][homo_col_idx[i]] == 1: #在同质性的边位置有连接
            adj_full_numpy[homo_row_idx[i]][homo_col_idx[i]] = 0  #去掉该同质性的边
            adj_full_numpy[homo_col_idx[i]][homo_row_idx[i]] = 0  #去掉对称边
            rm_homo_count = rm_homo_count + 1
            if rm_homo_count == rm_homo_budget:
                break
    print("Make Attack by Adding", add_hete_count, "Hete Edges and Dropping", rm_homo_count, "Homo Edges.")
    return adj_full_numpy

def aug(datasetname, adj_full_numpy, rm_hete_budget, add_homo_budget):
    #Step 1
    #在augmentation场景下 优先除去异质边来增强
    homo_row_idx, homo_col_idx, hete_row_idx, hete_col_idx = preload_idx(datasetname, "./")
    max_idx = np.max(hete_row_idx) if np.max(hete_row_idx) > np.max(hete_col_idx) else np.max(hete_col_idx)
    if max_idx <= (adj_full_numpy.shape[0] - 1000):
        print("No Test Nodes will be Affected in Removing hete Edges")
    else:
        print("Invalid Node Index")
        return
    rm_hete_count = 0
    print("Test1", np.count_nonzero(adj_full_numpy))
    for i in range(len(hete_row_idx)):
        if rm_hete_budget == 0:
            break
            #if hete_row_idx[i] <= hete_col_idx[i]:
            #    continue #如果这个位置在对角线或者对角线上面则不考虑
            #else:
        if adj_full_numpy[hete_row_idx[i]][hete_col_idx[i]] == 1: #在异质性的边位置有连接
            adj_full_numpy[hete_row_idx[i]][hete_col_idx[i]] = 0  #去掉该异质性的边
            adj_full_numpy[hete_col_idx[i]][hete_row_idx[i]] = 0  #去掉对称边
            rm_hete_count = rm_hete_count + 1
            #print("rm_hete_count:", rm_hete_count)
            if rm_hete_count == rm_hete_budget:
                break

    #Step 2
    #增加特定数量的同质性边【可能】会给精度带来正向收益
    max_idx = np.max(homo_row_idx) if np.max(homo_row_idx) > np.max(homo_col_idx) else np.max(homo_col_idx)
    if max_idx <= (adj_full_numpy.shape[0] - 1000):
        print("No Test Nodes will be Affected in Adding hete Edges")
    else:
        print("Invalid Node Index")
        return
    add_homo_count = 0
    for i in range(len(homo_row_idx)):
        #if homo_row_idx[i] <= homo_col_idx[i]:
        #    continue #如果这个位置在对角线或者对角线上面则不考虑
        #else:
        if add_homo_budget == 0:
            break
        if adj_full_numpy[homo_row_idx[i]][homo_col_idx[i]] == 0: #在同质性的边位置无连
            adj_full_numpy[homo_row_idx[i]][homo_col_idx[i]] = 1  #添加该同质性的边
            adj_full_numpy[homo_col_idx[i]][homo_row_idx[i]] = 1
            add_homo_count = add_homo_count + 1
            if add_homo_count == add_homo_budget:
                break
    print("Make Augmentation by Dropping", rm_hete_count, "Hete Edges and Adding", add_homo_count, "Homo Edges.")
    return adj_full_numpy
# Attention Please!
# How to use: 
# 1. full_size_metric_mt = compt_metric(dataset_name, adj_train_numpy)
# 2. adj_full_numpy = atk(adj_full_numpy, full_size_metric_mt, budget)
# 3. save adj_full_numpy as .graph file
# 4. train a GNN variant under the perturbed dataset


def cpt_component(adj):
    n_components, labels = csgraph.connected_components(adj)
    for i in range(n_components):
        component = np.where(labels == i)[0]    
    return n_components