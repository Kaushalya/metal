import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
from meta_learn.active_learner import ActiveLearner
from .utils import calculate_info_density


class CentralityActiveLearner(ActiveLearner):
    """
    Implementation of Active Graph Embedding (Cai et al., 2017)
    https://arxiv.org/abs/1705.05085

    Nodes are selected based on a linear combination of entropy, 
    PageRank centrality, and information density measured as the 
    euclidean distance to the nearest kmeans cluster.

    criteria = alpha * entropy + beta * information-density + 
    gamma * pagerank

    This code is adapted from the original implementation of AGE
    on https://github.com/vwz/AGE
    """

    def __init__(self, model, n_nodes, n_classes, n_features, oracle, query_strategy=None,
                 train_steps=100, learning_rate=0.1, momentum=0.9, debug=False,
                 batch_size=4,
                 device=None,
                 normalize_adj=True,
                 policy=None,
                 mc_dropout=False,
                 mc_dropout_iter=10,
                 weakly_supervised=False,
                 feature_meta_grad=False,
                 output_file_name=None,
                 gamma=0.,
                 optimizer=None):
        super(CentralityActiveLearner, self).__init__(model, n_nodes, n_classes,
                                                n_features, oracle, train_steps=train_steps,
                                                learning_rate=learning_rate, policy=policy,
                                                query_strategy=query_strategy,
                                                momentum=momentum, debug=debug, batch_size=batch_size,
                                                device=device, mc_dropout=mc_dropout,
                                                output_file_name=output_file_name,
                                                optimizer=optimizer)
        self.cen_perc = None
        self.basef = 0.9
        self.n_clusts = n_classes
        if policy in ('pagerank', 'degree'):
            self.gamma = 1
        else:
            self.gamma = gamma

    def calculate_centralities(self, adj):
        cenlist = []
        G = nx.from_numpy_matrix(adj.cpu().to_dense().numpy())
        if self.policy=='degree':
            cenlist.append(nx.degree_centrality(G))
        else:
            cenlist.append(nx.pagerank(G))
        n_cens = len(cenlist)
        L = len(cenlist[0])
        cenarray = np.zeros((n_cens, L))

        for i in range(n_cens):
            cenarray[i][list(cenlist[i].keys())] = list(cenlist[i].values())
        normcen = (cenarray.astype(float)-np.min(cenarray, axis=1)[:, None])/(
            np.max(cenarray, axis=1)-np.min(cenarray, axis=1))[:, None]
        normcen = normcen.squeeze(0)
        self.cen_perc = np.asarray([perc(normcen, i)
                                    for i in range(len(normcen))])

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids, epoch, query_loss=0.):
        if self.gamma == 0.:
            # gamma = np.random.beta(1, 1.005-self.basef**epoch)
            gamma = 0.5
        else:
            gamma = self.gamma

        alpha = beta = (1-gamma)/2

        if self.cen_perc is None and gamma > 0.:
            self.calculate_centralities(adj)
        
        if alpha > 0 or beta > 0:
            with torch.no_grad():
                logits = self.model.forward(adj, feature_matrix,
                                            param_dict=self.named_weights)

        # logits = self.forward_mc(adj, feature_matrix)
        if alpha > 0:
            node_entropy = self.calculate_entropy(
                torch.log_softmax(logits, dim=-1))
            node_entropy = node_entropy.cpu().numpy()
            ent_perc = np.asarray([perc(node_entropy, i)
                                for i in range(len(node_entropy))])
        if beta > 0:
            logits = logits.cpu().numpy()
            # kmeans = KMeans(n_clusters=self.n_clusts, random_state=0).fit(logits)
            # ed = euclidean_distances(logits, kmeans.cluster_centers_)
            # # the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is
            # ed_score = np.min(ed, axis=1)
            ed_perc = calculate_info_density(self.n_clusts, logits)

        criteria = np.zeros(self.n_nodes, dtype=np.float)

        if alpha > 0:
            criteria += alpha * ent_perc
        if beta > 0:
            criteria += beta * ed_perc
        if gamma > 0:
            criteria += gamma * self.cen_perc

        return torch.from_numpy(criteria).to(self.device)


#calculate the percentage of elements smaller than the k-th element
def perc(input, k): 
    return sum([1 if i else 0 for i in input < input[k]])/float(len(input))

#calculate the percentage of elements larger than the k-th element
def percd(input, k): 
    return sum([1 if i else 0 for i in input > input[k]])/float(len(input))
