import scipy as sp
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
import torch

import json
import logging
import os
import sys
import pickle as pkl
from networkx.readwrite import json_graph


def load_npz_dense(file_name):
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
    data = np.load(file_name, allow_pickle=True)
    adj_matrix = data['adj_data']
    attr_matrix = data['attr_data']
    labels = data['labels']

    return adj_matrix, attr_matrix, labels


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
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = csr_matrix((loader['attr_data'], loader['attr_indices'],
                                      loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            # attr_matrix = None
            # Use an identity matrix as the attribute matrix
            attr_matrix = csr_matrix(np.eye(adj_matrix._shape[0]))
        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(data_dir, dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(data_dir, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(data_dir, dataset_str))
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

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]
    labels = labels.argmax(1)
    
    return adj, features, labels


def preprocess_adj(adj, device):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    adj_ = adj + torch.eye(adj.shape[0], device=device)
    rowsum = torch.sum(adj_, dim=0)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5))
    adj_norm = torch.mm(torch.mm(adj_, degree_mat_inv_sqrt.transpose(1, 0)),
                        degree_mat_inv_sqrt)
    return adj_norm


def preprocess_sparse_adj(adj):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.sparse.coo_matrix(adj)
    rowsum = np.array(np.sum(adj, axis=0))
    degree_inv_sqrt = np.power(rowsum, -0.5).flatten()
    degree_mat_inv_sqrt = sp.sparse.diags(degree_inv_sqrt, 0)
    adj = sp.dot(sp.dot(adj, degree_mat_inv_sqrt.transpose()),
                        degree_mat_inv_sqrt)
    return adj.tocoo()


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    # reverse order to sort descending
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    logging.info("Selecting {n_components} largest connected components")
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None,
                                 random_state=None):
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
                                                   train_size=(
                                                       train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    if val_size > 0.:
        idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(
                                              train_size / (train_size + val_size)),
                                          test_size=(
                                              val_size / (train_size + val_size)),
                                          stratify=stratify)
    else:
        idx_train = idx_train_and_val
        idx_val = []
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def train_test_split_balanced(*arrays, labels, n_train_per_class, test_size, unlabeled_size,
                              stratify=None, random_state=None):
    """
    Splits the dataset into train and test in a way that each label
    has an equal number of samples.
    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train = []

    for i in np.unique(labels):
        labeled_ids = np.where(labels==i)[0]
        if random_state is not None:
            idx_train.extend(random_state.choice(labeled_ids, n_train_per_class))
        else:
            idx_train.extend(np.random.choice(labeled_ids, n_train_per_class))

    idx_val_and_test = np.setdiff1d(idx, idx_train)

    if stratify is not None:
        stratify = stratify[idx_val_and_test]
    if test_size > 0.:
        idx_val, idx_test = train_test_split(idx_val_and_test,
                                          random_state=random_state,
                                          train_size=(
                                              test_size / (test_size + unlabeled_size)),
                                          test_size=(
                                              unlabeled_size / (test_size + unlabeled_size)),
                                          stratify=stratify)
    else:
        idx_test = idx_val_and_test
        idx_val = []
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    
    logging.debug(f"train ids: {idx_train}")
    return result


def sparse_to_torch(mat, dtype='float'):
    """"
    Creates a torch sparse coo tensor from a given scipy coo matrix
    """

    values = mat.data
    indices = np.vstack((mat.row, mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sparse_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
