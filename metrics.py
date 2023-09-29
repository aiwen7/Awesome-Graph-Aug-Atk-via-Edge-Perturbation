import tensorflow.compat.v1 as tf
import numpy as np
import pickle as pkl
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components
tf.compat.v1.disable_eager_execution()


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

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

def load_file(file_path):
    with open(file_path, "rb") as f:
        file = pkl.load(f)
    return file

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
    label_mt_numpy = load_file("./data/ind." + dataset_name + ".ally") # .y仅仅是包括训练节点，.ally是训练与验证节点
    length = label_mt_numpy.shape[0] - 500
    print(length, "nodes are considered ...")     # 按顺序读取的节点，例如cora，从0到N号节点
    homo_mt = np.zeros((node_num, node_num))
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            
            elif (label_mt_numpy[i] == label_mt_numpy[j]).all():
                homo_mt[i][j] = 1
            else:
                homo_mt[i][j] = -1
    
    search_homo_idx = np.where(homo_mt == 1)
    homo_row_idx = search_homo_idx[0]
    homo_col_idx = search_homo_idx[1]
    homo_row_idx, homo_col_idx = random_shuffle(homo_row_idx, homo_col_idx)
    search_hete_idx = np.where(homo_mt == -1)
    hete_row_idx = search_hete_idx[0]
    hete_col_idx = search_hete_idx[1]
    hete_row_idx, hete_col_idx = random_shuffle(hete_row_idx, hete_col_idx)
    np.save("homo_row_idx.npy", homo_row_idx)
    np.save("homo_col_idx.npy", homo_col_idx)
    np.save("hete_row_idx.npy", hete_row_idx)
    np.save("hete_col_idx.npy", hete_col_idx)
    print("All Row && Col Index Saved.")
    return #homo_mt #所有元素要么是1要么是-1



def cpt_component(adj):
    n_components, labels = csgraph.connected_components(adj)
    for i in range(n_components):
        component = np.where(labels == i)[0]    
    return n_components

def preload_idx(path):
    homo_row_idx = np.load(path + "homo_row_idx.npy", allow_pickle=True)
    homo_col_idx = np.load(path + "homo_col_idx.npy", allow_pickle=True)
    hete_row_idx = np.load(path + "hete_row_idx.npy", allow_pickle=True)
    hete_col_idx = np.load(path + "hete_col_idx.npy", allow_pickle=True)
    return homo_row_idx, homo_col_idx, hete_row_idx, hete_col_idx

def atk(adj_full_numpy, add_hete_budget, rm_homo_budget):
    #Step 1
    #在attack场景下 优先添加异质边来进行攻击
    homo_row_idx, homo_col_idx, hete_row_idx, hete_col_idx = preload_idx("./")
    max_idx = np.max(hete_row_idx) if np.max(hete_row_idx) > np.max(hete_col_idx) else np.max(hete_col_idx)
    if max_idx <= (adj_full_numpy.shape[0] - 1000):
        print("No Test Nodes will be Affected in Adding hete Edges")
    else:
        print("Invalid Node Index")
        return
    add_hete_count = 0
    for i in range(len(hete_row_idx)):
        if hete_row_idx[i] <= hete_row_idx[i]:
            continue #如果这个位置在对角线或者对角线上面则不考虑
        else:
            if add_hete_budget == 0:
                break
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
        if homo_row_idx[i] <= hete_col_idx[i]:
            continue #如果这个位置在对角线或者对角线上面则不考虑
        else:
            if rm_homo_budget == 0:
                break
            if adj_full_numpy[homo_row_idx[i]][hete_col_idx[i]] == 1: #在同质性的边位置有连接
                adj_full_numpy[homo_row_idx[i]][hete_col_idx[i]] = 0  #去掉该同质性的边
                adj_full_numpy[hete_col_idx[i]][homo_row_idx[i]] = 0  #去掉对称边
                rm_homo_count = rm_homo_count + 1
                if rm_homo_count == rm_homo_budget:
                    break
    print("Make Attack by Adding", add_hete_count, "Hete Edges and Dropping", rm_homo_count, "Homo Edges.")
    return adj_full_numpy