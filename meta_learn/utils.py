import networkx as nx
import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def avg_shortest_path_length(graph, source_nodes, target_nodes):
    path_length = 0.
    n_paths = 0

    for source in source_nodes:
        for target in target_nodes:
            path_length += nx.shortest_path_length(graph, source=source, target=target)
            n_paths += 1

    return path_length/n_paths


def logit_mean(logits, dim: int):
    return torch.logsumexp(logits, dim=dim) - np.log(logits.shape[dim])


def entropy(logits, dim: int, weights=None):
    if weights is not None:
        return -torch.sum(torch.exp(logits) * logits * weights, dim=dim)
    return -torch.sum(torch.exp(logits) * logits, dim=dim)


def mutual_information(logits):
    sample_entropies = entropy(logits, dim=-1)
    entropy_mean = torch.mean(sample_entropies, dim=0)

    logits_mean = logit_mean(logits, dim=0)
    mean_entropy = entropy(logits_mean, dim=-1)

    mutual_info = mean_entropy - entropy_mean
    return mutual_info


#calculate the percentage of elements smaller than the k-th element
def perc(input, k): 
    return sum([1 if i else 0 for i in input < input[k]])/float(len(input))


#calculate the percentage of elements larger than the k-th element
def percd(input, k): 
    return sum([1 if i else 0 for i in input > input[k]])/float(len(input))


def calculate_info_density(n_clusters, features):
    '''
    Calculates information density metric
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    ed = euclidean_distances(features, kmeans.cluster_centers_)
    # the larger ed_score is, the far that node is away from cluster centers, 
    # the less representativeness the node is
    ed_score = np.min(ed, axis=1)
    ed_perc = np.asarray([percd(ed_score, i) for i in range(len(ed_score))])

    return ed_perc


def sgc_precompute(x, adj, degree, concat=False):
    for i in range(degree):
        x = torch.sparse.mm(adj, x)

    return x

def torch_to_graph(adj):
    return nx.from_numpy_array(adj.cpu().numpy())


def zero_grad(params):
    """
    Sets the gradient of a given set of weights (iterable or
    a dictionary) to zero.
    """
    if isinstance(params, dict):
        params = params.values()
    for param in params:
        if param.requires_grad and param.grad is not None:
            param.grad.zero_()


def get_model_weights(model):
    """
    Returnsa copy of the weights of a model as a dictionary
    (layer_name: wieght vector)
    """
    param_dict = dict()
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if model.layer_names and name in model.layer_names:
            param_dict[name] = param.clone().detach().requires_grad_(True)
    return param_dict

class GraphOracle(object):

    def __init__(self, labels):
        self.true_labels = labels

    def get_labels(self, id_list):
        return self.true_labels[id_list]
