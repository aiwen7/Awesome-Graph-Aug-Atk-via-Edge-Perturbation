from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy import linalg
from scipy.sparse import csr_matrix
import math
from numba import jit
import networkx as nx
import pickle as pkl
import sys
import utils


import cupy as cp
import cupyx.scipy.sparse as cusparse

class calculate:
    def __init__(self, adj, z_obs, u, X_mean, eig_vals, eig_vec, K, T, perturb_logs):

        # Adjacency matrix
        self.adj = adj.copy().tolil()
        #self.adj_train = adj_train
        #print("adj shape"*3)
        #print(adj.shape)
        #print("adj shape"*3)
        self.adj_no_selfloops = self.adj.copy()
        self.adj_no_selfloops.setdiag(0)
        self.adj_orig = self.adj.copy().tolil()
        self.u = u  # the node being attacked
        self.adj_preprocessed = utils.preprocess_graph(self.adj).tolil()
        # Number of nodes
        self.N = adj.shape[0]

        # Node attributes
        self.X_mean = X_mean

        self.eig_vals = eig_vals
        self.eig_vec = eig_vec

        #The order of graph filter K
        self.K = K

        #Top-T largest eigen-values/vectors selected
        self.T = T

        # Node labels
        self.z_obs = z_obs.copy()
        self.label_u = self.z_obs[self.u]
        self.K = np.max(self.z_obs)+1

        self.structure_perturbations = []

        self.potential_edges = []
        self.save_perturb_logs = perturb_logs

    def compute_log_likelihood(n, alpha, S_d, d_min):

        return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d


    def filter_singletons(edges, adj):

        degs = np.squeeze(np.array(np.sum(adj,0)))
        existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
        if existing_edges.size > 0:
            edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
        else:
            edge_degrees = degs[np.array(edges)] + 1

        zeros = edge_degrees == 0
        zeros_sum = zeros.sum(1)
        return zeros_sum == 0

    def filter_chisquare(ll_ratios, cutoff):
        
        return ll_ratios < cutoff

    def compute_alpha(n, S_d, d_min):

        return n / (S_d - n * np.log(d_min - 0.5)) + 1

    def update_Sx(S_old, n_old, d_old, d_new, d_min):


        old_in_range = d_old >= d_min
        new_in_range = d_new >= d_min

        d_old_in_range = np.multiply(d_old, old_in_range)
        d_new_in_range = np.multiply(d_new, new_in_range)

        new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
        new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

        return new_S_d, new_n


    def attack_model(self, n_perturbations, delta_cutoff=0.004):


            assert n_perturbations > 0, "need at least one perturbation"

            print("##### Starting attack #####")
            print("##### Attacking structure #####")
            print("##### Performing {} perturbations #####".format(n_perturbations))


            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.adj_orig.sum(0).A1
            current_degree_sequence = self.adj.sum(0).A1
            d_min = 2 #denotes the minimum degree a node needs to have to be considered in the power-law test
            S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = self.compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = self.compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

            # direct attack
            self.potential_edges = np.column_stack((np.tile(self.u, self.N-1), np.setdiff1d(np.arange(self.N), self.u)))

            self.potential_edges = self.potential_edges.astype("int32")
            
            scores = None
            for _ in range(n_perturbations):
                print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
                # Do not consider edges that, if removed, result in singleton edges in the graph.
                singleton_filter = self.filter_singletons(self.potential_edges, self.adj)
                filtered_edges = self.potential_edges[singleton_filter]

                # Update the values for the power law likelihood ratio test.
                deltas = 2 * (1 - self.adj[tuple(filtered_edges.T)].toarray()[0] )- 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
                new_S_d, new_n = self.update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = self.compute_alpha(new_n, new_S_d, d_min)
                new_ll = self.compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
                alphas_combined = self.compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = self.compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi_square cutoff value.
                powerlaw_filter = self.filter_chisquare(new_ratios, delta_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]

                print("-----filtered_edges----"*3)
                print(filtered_edges_final)
                print(len(filtered_edges_final))
                #exit()
                # Compute the struct scores for each potential edge as described in paper.
                struct_scores, scores_matrix = self.cal_scores(self.adj, self.X_mean, self.eig_vals, self.eig_vec, filtered_edges_final, K=self.K, T=self.T, lambda_method = "nosum")

                scores = struct_scores

                struct_scores = struct_scores.reshape(struct_scores.shape[0],1)

                best_edge_ix = struct_scores.argmax()
                best_edge_score = struct_scores.max()
                best_edge = filtered_edges_final[best_edge_ix]
                while (tuple(best_edge) in self.structure_perturbations):
                    struct_scores[best_edge_ix] = 0
                    best_edge_ix = struct_scores.argmax()
                    best_edge_score = struct_scores.max()
                    best_edge = filtered_edges_final[best_edge_ix]

                label_ori = self.adj[tuple(best_edge)]
                self.adj[tuple(best_edge)] = self.adj[tuple(best_edge[::-1])] = 1 - self.adj[tuple(best_edge)]

                with open(self.save_perturb_logs,"a+") as f:
                    f.write(str(best_edge) + ' ' + str(label_ori) + '\n')

                self.adj_preprocessed = self.preprocess_graph(self.adj)

                self.structure_perturbations.append(tuple(best_edge))

                # Update likelihood ratio test values
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]
            print('#'*40)   
            print(np.count_nonzero(scores_matrix))
            #print(filtered_edges)
            print(filtered_edges.shape)
            print(filtered_edges_final.shape)
            #print(type(scores_matrix))
            #print(scores_matrix.shape)
            print('#'*40)
    #@profile
    def cal_scores(result_matrix, A, X_mean, eig_vals, eig_vec, filtered_edges, K=2, T=128, lambda_method = "nosum"):
        #ndarray,scipy.sparse._arrays.csr_array,numpy.matrix,ndarray,ndarray,ndarray
        #pdb.set_trace()
        results = []
        A = A + sp.eye(A.shape[0])
        A[A > 1] = 1
        rowsum = A.sum(axis=1)
        A = sp.coo_matrix(A).tocsr() #numpy array
        A = cusparse.csr_matrix(A)
        D_min = rowsum.min()
        eig_vals = cp.asarray(eig_vals)
        eig_vec = cp.asarray(eig_vec)
        filtered_edges = cp.asarray(filtered_edges)
        X_mean = cp.asarray(X_mean)
        
        abs_V = len(eig_vals)
        tmp_k = T

        return_values = []

        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]
            eig_vals_res = cp.zeros(len(eig_vals))
            eig_vals_res = (1 - 2*A[filtered_edge[0], filtered_edge[1]]) * (2* eig_vec[filtered_edge[0],:] * eig_vec[filtered_edge[1],:] - eig_vals *
                                                                            ( cp.square(eig_vec[filtered_edge[0],:]) + cp.square(eig_vec[filtered_edge[1],:])))
            eig_vals_res = eig_vals + eig_vals_res

            if lambda_method == "sum":
                if K==1:
                    eig_vals_res =cp.abs(eig_vals_res / K) * (1/D_min)
                else:
                    for itr in range(1,K):
                        eig_vals_res = eig_vals_res + cp.power(eig_vals_res, itr+1)
                    eig_vals_res = cp.abs(eig_vals_res / K) * (1/D_min)
            else:
                eig_vals_res = cp.square((eig_vals_res + cp.ones(len(eig_vals_res))))
                eig_vals_res = cp.power(eig_vals_res, K)

            eig_vals_idx = cp.argsort(eig_vals_res)  # from small to large
            #idx = list(eig_vals_idx[:tmp_k])
            eig_vals_k_sum = eig_vals_res[eig_vals_idx[:tmp_k]].sum()
            # pdb.set_trace()
            u_k = eig_vec[:,eig_vals_idx[:tmp_k]]
            #u_k = eig_vec[:,idx]
            u_x_mean = u_k.T.dot(X_mean)
            return_values.append(eig_vals_k_sum * cp.square(cp.linalg.norm(u_x_mean)))

            print("The one_edge_version progress:%f%%" % (((j + 1) / (len(filtered_edges))) * 100), end='\r', flush=True)
        print("\n")
        #print(return_values[1])
        #print(len(return_values))
        #print(np.array(return_values))
        #print(type(return_values))
        #print(len(return_values))
        #print(len(filtered_edges))
        #exit()
    ################edit here################
    
        num_nodes = A.shape[0]
        return_values_array = cp.array(return_values)
        for j in range(len(filtered_edges)):
            filtered_edge = filtered_edges[j]            
            result_matrix[filtered_edge[0], filtered_edge[1]] = return_values_array[j]
            result_matrix[filtered_edge[1], filtered_edge[0]] = return_values_array[j]
        
    ################edit ends################
        return cp.array(return_values), result_matrix

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
    '''
    def load_data(dataset_str):
        def parse_index_file(filename):
            index = []
            for line in open(filename):
                index.append(int(line.strip()))
            return index
        def sample_mask(idx, l):
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

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
        
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

        #return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask
        #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        return adj, features_combined, labels[combined_index, :], y_train, y_val, y_test, train_mask, val_mask, test_mask
        '''
    
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
        #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        #return adj, features_combined, labels[combined_index, :], y_train, y_val, y_test, train_mask, val_mask, test_mask

    def filter_chisquare(ll_ratios, cutoff):
        return ll_ratios < cutoff
    
    def filter_singletons(edges, adj):
        degs = np.squeeze(np.array(np.sum(adj,0)))
        existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
        if existing_edges.size > 0:
            edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
        else:
            edge_degrees = degs[np.array(edges)] + 1

        zeros = edge_degrees == 0
        zeros_sum = zeros.sum(1)
        return zeros_sum == 0

    def filter_singletons(edges, adj):

        degs = np.squeeze(np.array(np.sum(adj,0)))
        existing_edges = np.squeeze(np.array(adj.tocsr()[tuple(edges.T)]))
        if existing_edges.size > 0:
            edge_degrees = degs[np.array(edges)] + 2*(1-existing_edges[:,None]) - 1
        else:
            edge_degrees = degs[np.array(edges)] + 1

        zeros = edge_degrees == 0
        zeros_sum = zeros.sum(1)
        return zeros_sum == 0
    def compute_alpha(n, S_d, d_min):
    
        return n / (S_d - n * np.log(d_min - 0.5)) + 1
    
    def update_Sx(S_old, n_old, d_old, d_new, d_min):

        old_in_range = d_old >= d_min
        new_in_range = d_new >= d_min

        d_old_in_range = np.multiply(d_old, old_in_range)
        d_new_in_range = np.multiply(d_new, new_in_range)

        new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
        new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)

        return new_S_d, new_n
    
    
def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components ( adj ) #Return the length-N array of each node's label in the connected components.
    component_sizes = np.bincount(component_indices) #Count number of occurrences of each value in array of non-negative ints.
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

