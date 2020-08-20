import csv
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.utils.data.sampler import WeightedRandomSampler
from meta_learn import metrics
from .utils import avg_shortest_path_length, torch_to_graph, zero_grad, entropy,\
    logit_mean, mutual_information
import logging

from collections import defaultdict


def calculate_metagrad_updated(learner: 'MetaActiveLearner', adj, feature_matrix,
                               labels, pred_probs, train_ids, unlabeled_ids,
                               epoch, meta_iter=30, alpha=0., beta=1.0):
    """"
    Return meta-gradient of the prediction entropy of
    unlabeled items

    1. Retrain the model with perturbed labels multiplied by mask m
    2. Calculate gradient of loss with respect to the mask m

    -- called from select_queries()
    """
    # TODO batch-mode selection - remove each selected item from unlabeled ids and
    # recalculate meta gradient
    labels_modified = torch.softmax(labels*(1+learner.label_changes), dim=1)
    # meta_grad = torch.zeros_like(labels_modified, device=learner.device)
    labels_1d = labels.argmax(1)
    named_weights_copy = learner.named_weights.copy()
    velocities_copy = learner.velocities.copy()

    for iter in range(meta_iter):
        zero_grad(learner.model, param_dict=named_weights_copy)
        preds = learner.model.forward(
            adj, feature_matrix, param_dict=named_weights_copy)
        # calculate cross-entropy loss between predictions and target
        loss = metrics.cross_entropy_loss(preds[unlabeled_ids],
                                            labels_modified[unlabeled_ids], weighted=False)
        logging.debug("epoch:{} train-loss = {}".format(iter, loss.data))
        grads = torch.autograd.grad(
            loss, named_weights_copy.values(), create_graph=True,
            retain_graph=True)
        velocities_copy = [(learner.momentum * v) +
                            grad for grad, v in zip(grads, velocities_copy)]
        current_params = [w - (learner.learning_rate * v) for w,
                            v in zip(named_weights_copy.values(), velocities_copy)]
        for key, param in zip(learner.named_weights.keys(), current_params):
            named_weights_copy[key] = param

    preds_perturbed = learner.model.forward(adj, feature_matrix,
                                                train_dropout=False,
                                                param_dict=named_weights_copy)
    unlabeled_mask = 1.0 - learner.label_changes[unlabeled_ids].sum(dim=1)
    meta_loss = alpha*F.cross_entropy(preds_perturbed[train_ids], labels_1d[train_ids])\
        + beta*(entropy(
            torch.log_softmax(preds_perturbed[unlabeled_ids], dim=1),
             dim=1)*unlabeled_mask).mean()
    # else:
    #     meta_loss = metrics.cross_entropy_loss(
    #         preds_perturbed[val_ids], labels[val_ids], weighted=False)

    zero_grad(learner.model, param_dict=named_weights_copy)
    # meta_grad_iter = torch.autograd.grad(
    #     meta_loss, self.label_changes, retain_graph=True)[0]
    meta_grad = -torch.autograd.grad(
        meta_loss, learner.label_changes, retain_graph=True)[0]
    meta_loss.detach()
    preds.detach()
    loss.detach()
    # max_pred_ind = torch.argmax(pred_probs, dim=1)

    # TODO is this needed?
    # for i in range(meta_grad.shape[0]):
    #     meta_grad[i, max_pred_ind[i]] = 0.
    # Calculate the expected meta-gradient

    return torch.sum(pred_probs*meta_grad, dim=1)


class ActiveLearner(nn.Module):
    def __init__(self, model, n_nodes, n_classes, n_features, oracle, query_strategy=None,
                 train_steps=100, learning_rate=0.1, momentum=0.9, debug=False,
                 batch_size=4,
                 batch_mode=False,
                 device=None,
                 normalize_adj=True,
                 policy="meta",
                 mc_dropout=False,
                 mc_dropout_iter=10,
                 output_file_name=None,
                 multilabel=False):
        super(ActiveLearner, self).__init__()
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.model = model
        self.oracle = oracle
        self.query_strategy = query_strategy
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.debug = debug
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.device = device
        self.normalize_adj = normalize_adj
        self.policy = policy
        self.mc_dropout = mc_dropout
        self.mc_dropout_iter = mc_dropout_iter
        self.named_weights = None
        self.output_file_name = output_file_name
        self.multilabel = multilabel

        if self.output_file_name is not None:
            with open(self.output_file_name, 'a') as output_file:
                results_writer = csv.writer(output_file)
                results_writer.writerow(['epoch', 'train size',
                                         'train accuracy', 'val accuracy', 'test accuracy',
                                         'macro-f1', 'micro-f1'])

        print(self)

    def forward(self, adj, feature_matrix, labels, train_ids, test_ids,
                unlabeled_ids, n_queries, return_performance=False, freq_analysis=False,
                distance_analysis=False):
        performance_dict = defaultdict(list)
        if freq_analysis:
            laplacian = torch.eye(self.n_nodes, device=self.device) - adj
            eig, eig_vec = torch.eig(laplacian, eigenvectors=True)
            # Order eigenvalues based on the real component
            eig_order = torch.argsort(eig[:, 0])
            eig_vec = eig_vec[:, eig_order]

        query_loss = 0.
        for i in range(n_queries):
            self.model.train()
            self._init_weights()
            query_ids, criteria = self.select_query_nodes(adj, feature_matrix,
                                                          labels, train_ids,
                                                          unlabeled_ids, i, query_loss=query_loss)
            train_ids = np.union1d(train_ids, query_ids)
            unlabeled_ids = np.setdiff1d(unlabeled_ids, query_ids)
            # Observe the label of nodes in query_ids
            labels[query_ids] = self.oracle.get_labels(query_ids)
            labels_1d = labels.argmax(dim=1)

            with torch.no_grad():
                logits = self.model(adj, feature_matrix)
                query_loss = F.cross_entropy(
                    logits[query_ids], labels_1d[query_ids])
                print("loss of query prediction: {:.3f}".format(query_loss))

            self.train_classifier(adj, feature_matrix, labels_1d, train_ids,
                                  create_graph=False)
            if return_performance:
                # scale criteria into [-1, 1]
                cmin = criteria.min()
                criteria = -2*(criteria-cmin)/(criteria.max() - cmin) + 1
                metrics = self._get_performance_metrics(
                    adj, feature_matrix, labels_1d, query_ids, train_ids, test_ids,
                    unlabeled_ids, i, freq_analysis, distance_analysis)
                for key, val in metrics.items():
                    performance_dict[key].append(val)
                # np.savetxt("results/freqs/freq_{}_{}.csv".format(self.policy, i),
                #                                 freq, delimiter=",")

        if return_performance:
            return (train_ids, unlabeled_ids, performance_dict)

        return (train_ids, unlabeled_ids)

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_id, epoch, **kwargs):
        raise NotImplementedError

    def calculate_entropy(self, logits):
        # Calculation of model uncertainty
        if len(logits.shape) == 3:
            logits = logit_mean(logits, dim=0)
        return entropy(logits, dim=1)

    def forward_mc(self, adj, feature_matrix, forward_iter=0):
        with torch.no_grad():
            if not forward_iter and self.mc_dropout:
                forward_iter = self.mc_dropout_iter
            preds_all = torch.zeros(
                (forward_iter, self.n_nodes, self.n_classes),
                device=self.device)

            for iter in range(forward_iter):
                preds_all[iter] = self.model(
                    adj, feature_matrix, mc_dropout=self.mc_dropout == 1, param_dict=self.named_weights)

        return torch.log_softmax(preds_all, dim=-1)

    def _get_performance_metrics(self, adj, feature_matrix, labels, query_ids, train_ids,
                                 test_ids, unlabeled_ids, iter_id, freq_analysis,
                                 distance_analysis):
        metric_dict = {}
        self.model.eval()
        with torch.no_grad():
            preds = self.model(adj, feature_matrix,
                               param_dict=self.named_weights)
            metric_dict['train-accuracy'] = metrics.accuracy(preds[train_ids],
                                                             self.oracle.get_labels(train_ids))
            metric_dict['test-accuracy'] = metrics.accuracy(
                preds[test_ids], self.oracle.get_labels(test_ids))
            metric_dict['unlabeled-accuracy'] = metrics.accuracy(
                preds[unlabeled_ids], self.oracle.get_labels(unlabeled_ids))
            metric_dict['macro-f1'] = metrics.f1_score(
                preds[test_ids], self.oracle.get_labels(test_ids), average='macro')
            metric_dict['micro-f1'] = metrics.f1_score(
                preds[test_ids], self.oracle.get_labels(test_ids), average='micro')
            print("train set size: {}".format(train_ids.shape[0]))
            summary_str = ", ".join(["{} = {:.3f}".format(x, y)
                                     for x, y in metric_dict.items()])
            print("perturbation:{} ".format(iter_id+1) + summary_str)

            # if freq_analysis:
            #     # calculation of Rayleigh quotient
            #     p = torch.mm(torch.mm(criteria.view(1, -1),
            #                         laplacian), criteria.view(-1, 1))
            #     q = criteria.dot(criteria)
            #     rayleigh_q = p/q
            #     freq = torch.mm(eig_vec.transpose(1,0), criteria.view((-1, 1)))
            #     print("Rayleigh quotient = {:.3f}".format(rayleigh_q.item()))
            #     metrics['rayleigh-quotient'] = rayleigh_q

            if distance_analysis:
                # calculate average shortest path lengths to training nodes
                # with the same label and different labels
                query_labels = labels[query_ids].unique()
                graph = torch_to_graph(adj)
                same_dist = 0.
                dif_dist = 0.
                n_same = 0
                n_dif = 0
                train_mask = torch.zeros(labels.shape, dtype=torch.bool,
                                         device=self.device)
                train_mask[train_ids] = True

                for qid in query_ids:
                    for label in range(self.n_classes):
                        label_nodes = (labels == label)
                        label_nodes = torch.nonzero(
                            label_nodes & train_mask, as_tuple=False).view(-1)
                        label_nodes = label_nodes.cpu().numpy()
                        if len(label_nodes) == 0:
                            continue
                        avg_path_train = avg_shortest_path_length(graph,
                                                                  [qid], label_nodes)
                        if labels[qid] == label:
                            same_dist += avg_path_train
                            n_same += 1
                        else:
                            dif_dist += avg_path_train
                            n_dif += 1
                avg_same_label_dist = same_dist/n_same
                avg_dif_label_dist = dif_dist/n_dif
                metric_dict['avg_sp_same_label'] = avg_same_label_dist
                metric_dict['avg_sp_dif_label'] = avg_dif_label_dist
                print("Average shortest path to nodes with same label {:.2f}".format(
                    avg_same_label_dist))
                print("Average shortest path to nodes with a different label {:.2f}".format(
                    avg_dif_label_dist))

        return metric_dict

    def _init_weights(self):
        if self.named_weights is None:
            self.named_weights = self._get_named_param_dict(
                self.model.named_parameters())
            self.velocities = [torch.zeros_like(
                w) for w in self.named_weights.values()]

        for i, key in enumerate(self.named_weights.keys()):
            self.named_weights[key] = self.named_weights[key].detach()
            self.named_weights[key].requires_grad = True
            self.velocities[i] = self.velocities[i].detach()

    def train_classifier(self, adj, feature_matrix, labels, train_ids, create_graph=False):

        for iter in range(self.train_steps):
            zero_grad(self.model, param_dict=self.named_weights)
            preds = self.model.forward(
                adj, feature_matrix, param_dict=self.named_weights)
            # loss = F.binary_cross_entropy_with_logits(preds[train_ids], labels[train_ids])
            loss = F.cross_entropy(preds[train_ids], labels[train_ids])
            if self.debug:
                print("epoch:{} train-loss = {}".format(iter, loss.data))
            # self.model.zero_grad(param_dict=self.named_weights)
            grads = torch.autograd.grad(
                loss, self.named_weights.values(), create_graph=create_graph)
            self.velocities = [(self.momentum * v) +
                               grad for grad, v in zip(grads, self.velocities)]
            current_params = [w - (self.learning_rate * v) for w,
                              v in zip(self.named_weights.values(), self.velocities)]
            self.set_weights(current_params)

    def _get_named_param_dict(self, params):
        param_dict = dict()
        for name, param in params:
            name = name.split('.')[0]
            if name in self.model.layer_names:
                param_dict[name] = param
        return param_dict

    def set_weights(self, params):
        for key, param in zip(self.named_weights.keys(), params):
            self.named_weights[key] = param

    def __str__(self):
        return '*** Active Learner ***\n policy: {}\n learning rate: {}\
            \n batch_size: {}'.format(self.policy, self.learning_rate, self.batch_size)


class RandomActiveLearner(ActiveLearner):

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids, epoch, **kwargs):
        unlabeled_mask = torch.zeros(self.n_nodes, dtype=torch.float32,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = 1
        criteria = torch.rand(self.n_nodes, device=self.device)
        criteria *= unlabeled_mask
        return np.random.choice(unlabeled_ids, self.batch_size,
                                replace=False), criteria


class BALDActiveLearner(ActiveLearner):
    """
    Implementation of the paper "Deep Bayesian Active Learning with Image Data"
    by Yarin Gal et al. (2017)
    """

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids, epoch, **kwargs):
        unlabeled_mask = torch.zeros(self.n_nodes, dtype=torch.float32,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = 1
        logits = self.forward_mc(adj, feature_matrix)
        criteria = mutual_information(logits)
        _, query_ids = torch.topk(
            criteria[unlabeled_ids], k=self.batch_size, largest=True)
        query_ids = query_ids.cpu().numpy()
        query_ids = unlabeled_ids[query_ids]

        return query_ids, criteria


class MetaActiveLearner(ActiveLearner):
    def __init__(self, model, n_nodes, n_classes, n_features, oracle, query_strategy=None,
                 train_steps=100, learning_rate=0.1, momentum=0.9, debug=False,
                 batch_size=4,
                 batch_mode=False,
                 device=None,
                 normalize_adj=True,
                 policy="meta",
                 mc_dropout=False,
                 mc_dropout_iter=10,
                 self_supervised=False,
                 feature_meta_grad=False,
                 output_file_name=None,
                 approx_meta_grad=False,
                 multi_meta_grad=False,
                 use_mutual_info=False):
        super(MetaActiveLearner, self).__init__(model, n_nodes, n_classes,
                                                n_features, oracle, train_steps=train_steps,
                                                learning_rate=learning_rate, policy=policy,
                                                momentum=momentum, debug=debug, batch_size=batch_size,
                                                batch_mode=batch_mode,
                                                device=device, mc_dropout=mc_dropout,
                                                output_file_name=output_file_name)
        # self.multi_meta_grad = multi_meta_grad
        self.mc_dropout_iter = mc_dropout_iter
        self.self_supervised = self_supervised
        # self.feature_meta_grad = feature_meta_grad
        self.meta_learning_rate = 1e-1
        # self.approx_meta_grad = approx_meta_grad
        self.use_mutual_info = use_mutual_info
        # TODO load this from a config
        self.meta_grad_type = 'masked'

        if self.meta_grad_type=='feature':
            self.feat_changes = nn.Parameter(
                torch.zeros(size=(n_nodes, n_features)))
            nn.init.xavier_normal_(self.feat_changes.data, gain=0.1)
        else:
            self.label_changes = nn.Parameter(
                torch.zeros(size=(n_nodes, n_classes)))
            nn.init.uniform_(self.label_changes.data, 0.1, 0.5)
            # nn.init.constant_(self.label_changes.data, 0.5)
        self.named_weights = None

        print(self)

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids, unlabeled_ids, epoch,
                           **kwargs):
        node_entropy = torch.ones((self.n_nodes), dtype=torch.float32,
                                  device=self.device)
        node_meta_grad = torch.ones_like(node_entropy)
        unlabeled_mask = torch.zeros(node_entropy.shape, dtype=torch.bool,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        query_ids = None
        # if self.policy in ('entropy', 'both') or self.policy=='':
        logits = self.forward_mc(adj, feature_matrix)
        pred_probs = torch.exp(logits).mean(dim=0)
        if self.policy in ('entropy', 'both'):
            node_entropy = self.calculate_entropy(logits)
        alpha = 1.0
        if self.use_mutual_info:
            mutual_info = mutual_information(logits)
            mutual_info = (mutual_info - mutual_info.min()) / \
                (mutual_info.max() - mutual_info.min())
            alpha = 0.8**kwargs['query_loss']
        # with torch.no_grad():
        #     preds = self.model.forward(adj, feature_matrix,
        #                                 param_dict=self.named_weights)
        #     pred_probs = torch.softmax(preds, dim=1)
        if self.policy == 'meta' or self.policy == 'both':
            # Uses a set of k nodes from the unlabeled set as the validation
            # set for computing the meta-gradient.

            if self.meta_grad_type=='feature':
                node_meta_grad = self._calculate_node_feature_meta_grad(adj, feature_matrix,
                                                                        labels, pred_probs,
                                                                        train_ids,
                                                                        unlabeled_ids)
            elif self.meta_grad_type=='multi':
                # average entropy of the model is used to weight the entropy term in the loss function.
                # avg_entropy = node_entropy[unlabeled_ids].mean()
                # TODO find a better way to define gamma from average entropy
                # gamma2 = 1. - torch.exp(-2*avg_entropy)
                # print("average node entropy = {:.2f}".format(avg_entropy))
                # gamma = 1.005-0.9**epoch
                if self.batch_size > 1 and self.batch_mode:
                    # selection_size = 1
                    beta = 1.0
                    query_ids = []

                    # TODO use features of last layer for calculating distance
                    meta_grad = self._calculate_node_multi_label_meta_grad(
                        adj, feature_matrix, labels, pred_probs, train_ids, unlabeled_ids, gamma=0.,
                        meta_iter=50)
                    criteria = meta_grad.sum(1)
                    # stores sum of euclidean distances to currently chosen nodes
                    edist = torch.zeros_like(criteria)

                    for idx in range(self.batch_size):
                        if idx > 0:
                            criteria += beta * edist/idx
                        best_node = (
                            criteria*unlabeled_mask.float()).argmax()
                        unlabeled_mask[best_node] = False
                        edist += torch.norm(meta_grad -
                                            meta_grad[best_node], 2, dim=1)
                        query_ids.append(best_node.item())
                else:
                    node_meta_grad = self._calculate_node_multi_label_meta_grad(
                        adj, feature_matrix, labels, pred_probs, train_ids, unlabeled_ids, gamma=0.,
                        meta_iter=50)
            elif self.meta_grad_type=='approx':
            # This is the default mode
                meta_iter = 100
                if self.batch_size>1:
                    meta_iter = max(100*self.batch_size, 500)
                if self.batch_size > 1 and self.batch_mode:
                    beta = 0.1
                    query_ids = []

                    meta_grad = self._calculate_node_label_approx_meta_grad(
                        adj, feature_matrix, labels, pred_probs, train_ids, unlabeled_ids,
                        meta_iter=meta_iter)
                    node_meta_grad = meta_grad.sum(1)
                    node_meta_grad = (node_meta_grad - node_meta_grad.min()) / \
                        (node_meta_grad.max() - node_meta_grad.min())
                    # stores sum of euclidean distances to currently chosen nodes
                    edist = torch.zeros_like(node_meta_grad)
                    if alpha < 1:
                        node_meta_grad = alpha * node_meta_grad + \
                            (1-alpha) * mutual_info

                    criteria = node_meta_grad

                    for idx in range(self.batch_size):
                        unlabeled_iter = np.setdiff1d(
                            unlabeled_ids, query_ids)
                        criteria = node_meta_grad
                        if idx > 0:
                            norm_dist = edist/idx
                            norm_dist = (
                                norm_dist-norm_dist.min())/(norm_dist.max()-norm_dist.min())
                            criteria += beta * norm_dist

                        best_node = unlabeled_iter[criteria[unlabeled_iter].argmax(
                        ).item()]
                        # best_node = (
                        #    criteria*unlabeled_mask.float()).argmax()
                        unlabeled_mask[best_node] = False
                        edist += torch.norm(pred_probs -
                                            pred_probs[best_node], 2, dim=1)
                        query_ids.append(best_node)
                else:
                    node_meta_grad = self._calculate_node_label_approx_meta_grad(
                        adj, feature_matrix, labels, pred_probs, train_ids, unlabeled_ids,
                        meta_iter=meta_iter)
            elif self.meta_grad_type == 'masked':
                alpha = min(0., 0.5-0.95**epoch)

                if self.batch_mode and self.batch_size > 1:
                    query_ids = list()
                    # remove selected item from unlabeled items and calculate
                    # meta gradients
                    # TODO needs a better way to remove items from unlabaled_ids
                    for b_i in range(self.batch_size):
                        node_meta_grad = calculate_metagrad_updated(self, adj, feature_matrix, labels,
                                                                    pred_probs, train_ids, unlabeled_ids, epoch,
                                                                    alpha=alpha)
                        best_node_id = torch.argmax(node_meta_grad*unlabeled_mask.float()).item()
                        # best_node_id = unlabeled_ids[node_meta_grad[unlabeled_ids].argmax(
                        # ).item()]
                        query_ids.append(best_node_id)
                        unlabeled_mask[best_node_id] = False
                        unlabeled_ids = np.setdiff1d(unlabeled_ids, query_ids)
                else:
                    node_meta_grad = calculate_metagrad_updated(self, adj, feature_matrix, labels,
                                                                pred_probs, train_ids, unlabeled_ids, epoch,
                                                                alpha=alpha)
            else:
                node_meta_grad = self._calculate_node_label_meta_grad(
                    adj, feature_matrix, labels, pred_probs, train_ids, unlabeled_ids)

        if self.policy == 'meta':
            criteria = node_meta_grad
        else:
            criteria = node_entropy * node_meta_grad

        criteria *= unlabeled_mask.float()

        # if self.thompson_sampling:
        # criteria -= criteria.min()
        # sampled_ids = list(WeightedRandomSampler(criteria, self.batch_size, replacement=False))

        if query_ids is None:
            _, query_ids = criteria.topk(self.batch_size, largest=True)
            # query_ids = query_ids[:self.batch_size]
            query_ids = query_ids.cpu().numpy()

        return query_ids, criteria

    def _calculate_node_label_meta_grad(self, adj, feature_matrix,
                                        labels, pred_probs, train_ids,
                                        unlabeled_ids,
                                        meta_iter=20):
        # alpha = 1e-4
        labels_modified = torch.softmax(labels*(1+self.label_changes), dim=1)
        # labels_modified = labels_modified/labels_modified.norm(dim=1)
        # labels_modified = torch.softmax(labels+self.label_changes, dim=1)
        named_weights_copy = self.named_weights.copy()
        velocities_copy = self.velocities.copy()
        zero_grad(self.model, param_dict=named_weights_copy)

        # TODO use a random set of unlabeled_ids as val_ids
        for iter in range(meta_iter):
            # zero_grad(self.model, param_dict=named_weights_copy)
            preds = self.model.forward(
                adj, feature_matrix, param_dict=named_weights_copy, train_dropout=False)
            # weakly supervised loss
            # loss = F.binary_cross_entropy_with_logits(preds[unlabeled_ids], labels_modified[unlabeled_ids])

            # calculate cross-entropy loss between predictions and target
            loss = metrics.cross_entropy_loss(preds[unlabeled_ids],
                                              labels_modified[unlabeled_ids])
            if self.debug:
                print("epoch:{} meta train-loss = {}".format(iter, loss.data))
            # self.model.zero_grad(param_dict=self.named_weights)
            grads = torch.autograd.grad(
                loss, named_weights_copy.values(), create_graph=True)
            velocities_copy = [(self.momentum * v) +
                               grad for grad, v in zip(grads, velocities_copy)]
            current_params = [w - (self.meta_learning_rate * v) for w,
                              v in zip(named_weights_copy.values(), velocities_copy)]
            for key, param in zip(self.named_weights.keys(), current_params):
                named_weights_copy[key] = param

        preds_perturbed = self.model.forward(
            adj, feature_matrix, train_dropout=False, param_dict=named_weights_copy)

        labels_1d = labels.argmax(dim=1)
        alpha = 1.
        if self.self_supervised:
            # meta_loss = metrics.cross_entropy_loss(
            #     preds_perturbed[unlabeled_ids], labels[unlabeled_ids])
            meta_loss = metrics.cross_entropy_loss(
                preds_perturbed[val_ids], labels[val_ids])
        else:
            meta_loss = F.cross_entropy(
                preds_perturbed[val_ids], labels_1d[val_ids])
        self.model.zero_grad()
        meta_grad = torch.autograd.grad(
            meta_loss, self.label_changes, retain_graph=True)
        meta_loss.detach()
        preds.detach()
        label_meta_grad = meta_grad[0]

        # if self.self_supervised:
        label_meta_grad *= -1
        max_pred_ind = torch.argmax(pred_probs, dim=1)

        for i in range(label_meta_grad.shape[0]):
            label_meta_grad[i, max_pred_ind[i]] = 0.
        # Calculate the expected meta-gradient
        node_meta_grad = torch.sum(pred_probs*label_meta_grad, dim=1)

        # ignore the current prediction probabilities
        # node_meta_grad = torch.sum(label_meta_grad, dim=1)

        return node_meta_grad

    def _calculate_node_label_approx_meta_grad(self, adj, feature_matrix,
                                               labels, pred_probs, train_ids,
                                               unlabeled_ids, current_batch=None, gamma=0.,
                                               meta_iter=10):
        labels_modified = torch.softmax(labels*(1+self.label_changes), dim=1)
        # labels_modified = torch.softmax(labels+self.label_changes, dim=1)
        # named_weights_copy = self.named_weights.copy()
        # velocities_copy = self.velocities.copy()
        meta_grad = torch.zeros_like(labels_modified, device=self.device)
        n_samples = 0.1 * torch.ones((self.n_nodes), dtype=torch.float32,
                                     device=self.device)
        unlabeled_mask = torch.zeros((self.n_nodes), dtype=torch.bool,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        labels_1d = labels.argmax(1)
        val_share = 0.5
        n_mc_iter = 2
        abs_grad = False
        discriminative_loss = False

        for iter in range(meta_iter):
            torch.nn.init.uniform_(self.label_changes, a=0.2, b=1.0)
            labels_modified = torch.softmax(
                labels*(1+self.label_changes), dim=1)

            named_weights_copy = self.named_weights.copy()
            velocities_copy = self.velocities.copy()

            # Randomly select a set of unlabeled nodes as the validation set.
            # This validation set is used to calculate the meta-gradient.
            k_ = int(val_share * unlabeled_ids.shape[0])
            val_ids = np.random.choice(unlabeled_ids, k_)
            val_mask = torch.zeros_like(unlabeled_mask)
            val_mask[val_ids] = True
            unlabeled_ids_iter = torch.where(unlabeled_mask & ~val_mask)[0]
            # unlabeled_mask = unlabeled_mask & ~val_mask

            for mc_iter in range(n_mc_iter):
                zero_grad(self.model, param_dict=named_weights_copy)
                preds = self.model.forward(
                    adj, feature_matrix, param_dict=named_weights_copy)
                # calculate cross-entropy loss between predictions and target
                loss = metrics.cross_entropy_loss(preds[unlabeled_ids_iter],
                                                  labels_modified[unlabeled_ids_iter], weighted=False)
                logging.debug("epoch:{} train-loss = {}".format(iter, loss.data))
                grads = torch.autograd.grad(
                    loss, named_weights_copy.values(), create_graph=True,
                    retain_graph=True)
                velocities_copy = [(self.momentum * v) +
                                   grad for grad, v in zip(grads, velocities_copy)]
                current_params = [w - (self.learning_rate * v) for w,
                                  v in zip(named_weights_copy.values(), velocities_copy)]
                for key, param in zip(self.named_weights.keys(), current_params):
                    named_weights_copy[key] = param

            preds_perturbed = self.model.forward(adj, feature_matrix,
                                                 train_dropout=False,
                                                 param_dict=named_weights_copy)

            # TODO Try entropy of predictions of unlabeled items instead of
            # meta loss
            if discriminative_loss:
                alpha = 0.1
                beta = 0.1
                meta_loss = alpha*F.cross_entropy(preds_perturbed[train_ids], labels_1d[train_ids]) + entropy(
                    torch.log_softmax(preds_perturbed[val_ids], dim=1), dim=1).mean() + beta*entropy(
                        torch.log_softmax(preds_perturbed[unlabeled_ids], dim=1), dim=1).mean()
            else:
                meta_loss = metrics.cross_entropy_loss(
                    preds_perturbed[val_ids], labels[val_ids], weighted=False)

            # meta_loss = metrics.focal_loss(preds_perturbed[val_ids], labels[val_ids], gamma=0.5)
            # meta_loss = 0.8 * meta_loss- 0.2 *  F.cross_entropy(
            #     preds_perturbed[train_ids], labels_1d[train_ids])
            zero_grad(self.model, param_dict=named_weights_copy)
            # meta_grad_iter = torch.autograd.grad(
            #     meta_loss, self.label_changes, retain_graph=True)[0]
            # TODO considering only the magnitude of gradients
            if abs_grad:
                meta_grad += torch.autograd.grad(
                    meta_loss, self.label_changes, retain_graph=True)[0]**2
            else:
                meta_grad += torch.autograd.grad(
                    meta_loss, self.label_changes, retain_graph=True)[0]
            n_samples[unlabeled_ids_iter] += 1.
            meta_loss.detach()
            preds.detach()
            loss.detach()

        # TODO divide by n_smaples, if samples are biased
        meta_grad /= n_samples.view((-1, 1))
        if not abs_grad:
            meta_grad *= -1
        max_pred_ind = torch.argmax(pred_probs, dim=1)

        # TODO is this needed?
        # for i in range(meta_grad.shape[0]):
        #     meta_grad[i, max_pred_ind[i]] = 0.
        # Calculate the expected meta-gradient
        if self.batch_size > 1 and self.batch_mode:
            return pred_probs*meta_grad
        return torch.sum(pred_probs*meta_grad, dim=1)

    def _calculate_node_multi_label_meta_grad(self, adj, feature_matrix,
                                              labels, pred_probs, train_ids,
                                              unlabeled_ids, gamma=0.,
                                              meta_iter=10):
        labels_1d = labels.argmax(dim=1)
        # label_meta_grads = torch.zeros_like(pred_probs, device=self.device)
        meta_grad = torch.zeros_like(pred_probs, device=self.device)
        unlabeled_mask = torch.zeros((self.n_nodes), dtype=torch.bool,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        val_share = 0.5

        for label_i in range(self.n_classes):
            column_mask = torch.zeros(
                labels.shape, dtype=torch.float32, device=self.device)
            column_mask[:, label_i] = 1
            # labels_modified = labels+self.label_changes*column_mask
            # multiplicative perturbations are added to each class.
            labels_modified = F.softmax(
                labels*(1+self.label_changes)*column_mask, dim=1)
            # named_weights_copy = self.named_weights.copy()
            # velocities_copy = self.velocities.copy()
            # keeps the count how many times a node has been chosen for the calidation set.
            n_samples = 0.1 * torch.ones((self.n_nodes), dtype=torch.float32,
                                         device=self.device)
            for iter in range(meta_iter):
                named_weights_copy = self.named_weights.copy()
                velocities_copy = self.velocities.copy()

                # Randomly select a set of nodes as the validation set.
                # This validation set is used to calculate the meta-gradient.
                k_ = int(val_share * unlabeled_ids.shape[0])
                rstate = np.random.RandomState(1234+meta_iter)
                val_ids = rstate.choice(unlabeled_ids, k_)
                val_mask = torch.zeros_like(unlabeled_mask)
                val_mask[val_ids] = True
                unlabeled_ids_iter = torch.where(unlabeled_mask & ~val_mask)[0]

                # TODO make this better
                mc_iter = 2

                for _ in range(mc_iter):
                    zero_grad(self.model, param_dict=named_weights_copy)
                    preds = self.model.forward(
                        adj, feature_matrix, param_dict=named_weights_copy)
                    # supervised loss on the training set and perturbed labels
                    loss = metrics.cross_entropy_loss(
                        preds[unlabeled_ids_iter], labels_modified[unlabeled_ids_iter]) + F.cross_entropy(preds[train_ids], labels_1d[train_ids])
                    if self.debug:
                        print("epoch:{} train-loss = {}".format(iter, loss.data))
                    grads = torch.autograd.grad(
                        loss, named_weights_copy.values(), create_graph=True)
                    velocities_copy = [(self.momentum * v) +
                                       grad for grad, v in zip(grads, velocities_copy)]
                    current_params = [w - (self.learning_rate * v) for w,
                                      v in zip(named_weights_copy.values(), velocities_copy)]
                    for key, param in zip(self.named_weights.keys(), current_params):
                        named_weights_copy[key] = param

                preds_perturbed = self.model.forward(
                    adj, feature_matrix, train_dropout=False, param_dict=named_weights_copy)

                meta_loss = (1-gamma) * metrics.cross_entropy_loss(
                    preds_perturbed[val_ids], labels[val_ids], weighted=True) + gamma * metrics.cross_entropy_loss(
                    preds_perturbed[val_ids], F.softmax(preds_perturbed[val_ids], dim=1))
                zero_grad(self.model, param_dict=named_weights_copy)
                meta_grad_iter = torch.autograd.grad(
                    meta_loss, self.label_changes, retain_graph=True)[0]
                meta_grad += meta_grad_iter
                n_samples[unlabeled_ids_iter] += 1.
                meta_loss.detach()
                preds.detach()
                loss.detach()
                # for key, value in named_weights_copy.items():
                #    val = value.detach()
                #    val.requires_grad = True
                #    named_weights_copy[key] = val

            # gamma = 0.5
            # meta_loss = metrics.cross_entropy_loss(
            #     preds_perturbed[val_ids], F.softmax(preds_perturbed[val_ids], dim=1))
            # meta_loss = metrics.cross_entropy_loss(preds_perturbed[val_ids], labels[val_ids])
            # meta_loss = gamma * meta_loss - \
            #     (1-gamma) * \
            #     F.cross_entropy(
            #         preds_perturbed[train_ids], labels_1d[train_ids])
            # self.model.zero_grad()
            # meta_grad = torch.autograd.grad(
            #     meta_loss, self.label_changes, retain_graph=True)[0]
            # meta_loss.detach()
            # preds.detach()
            meta_grad[:, label_i] = meta_grad[:, label_i]/n_samples

        max_pred_ind = torch.argmax(pred_probs, dim=1)
        # TODO make this step more efficient
        for i in range(meta_grad.shape[0]):
            meta_grad[i, max_pred_ind[i]] = 0.
        # Calculate the expected meta-gradeint
        if self.self_supervised:
            meta_grad *= -1
        # meta_grad = torch.sum(pred_probs*meta_grad, dim=1)

        # ignore the current prediction probabilities
        # node_meta_grad = torch.sum(meta_grad, dim=1)

        if self.batch_size > 1 and self.batch_mode:
            return pred_probs*meta_grad
        return torch.sum(pred_probs*meta_grad, dim=1)

    def _calculate_node_label_batch_meta_grad(self, adj, feature_matrix,
                                              labels, pred_probs, train_ids,
                                              unlabeled_ids, gamma=0.,
                                              meta_iter=5):
        """
        Meta learning for batch-mode active learning
        """
        labels_1d = labels.argmax(dim=1)
        # label_meta_grads = torch.zeros_like(pred_probs, device=self.device)
        meta_grad = torch.zeros_like(pred_probs, device=self.device)
        unlabeled_mask = torch.zeros((self.n_nodes), dtype=torch.bool,
                                     device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        val_share = 0.1
        current_batch = []
        n_samples = 0.1 * torch.ones((self.n_nodes), dtype=torch.float32,
                                     device=self.device)

        for k in self.batch_size:
            # Randomly select a set of nodes as the validation set.
            # This validation set is used to calculate the meta-gradient.
            k_ = int(val_share * unlabeled_ids.shape[0])
            val_ids = np.random.choice(unlabeled_ids, k_)
            val_mask = torch.zeros_like(unlabeled_mask)
            val_mask[val_ids] = True
            unlabeled_ids_iter = torch.where(unlabeled_mask & ~val_mask)[0]
            n_samples[unlabeled_ids_iter] += 1.

            for label_i in range(self.n_classes):
                column_mask = torch.zeros(
                    labels.shape, dtype=torch.float32, device=self.device)
                column_mask[:, label_i] = 1
                # labels_modified = labels+self.label_changes*column_mask
                # multiplicative perturbations are added to each class.
                labels_modified = F.softmax(
                    labels*(1+self.label_changes)*column_mask, dim=1)
                named_weights_copy = self.named_weights.copy()
                velocities_copy = self.velocities.copy()
                # keeps the count how many times a node has been chosen for the calidation set.

                # np.random.seed(1)
                for iter in range(meta_iter):
                    # Randomly select a set of nodes as the validation set.
                    # This validation set is used to calculate the meta-gradient.
                    # k_ = int(val_share * unlabeled_ids.shape[0])
                    # val_ids = np.random.choice(unlabeled_ids, k_)
                    # val_mask = torch.zeros_like(unlabeled_mask)
                    # val_mask[val_ids] = True
                    # unlabeled_ids_iter = torch.where(unlabeled_mask & ~val_mask)[0]

                    zero_grad(self.model, param_dict=named_weights_copy)
                    preds = self.model.forward(
                        adj, feature_matrix, param_dict=named_weights_copy)
                    # supervised loss on the training set and perturbed labels
                    loss = metrics.cross_entropy_loss(
                        preds[unlabeled_ids_iter], labels_modified[unlabeled_ids_iter]) + F.cross_entropy(preds[train_ids], labels_1d[train_ids])
                    if self.debug:
                        print("epoch:{} train-loss = {}".format(iter, loss.data))
                    grads = torch.autograd.grad(
                        loss, named_weights_copy.values(), create_graph=True)
                    velocities_copy = [(self.momentum * v) +
                                       grad for grad, v in zip(grads, velocities_copy)]
                    current_params = [w - (self.learning_rate * v) for w,
                                      v in zip(named_weights_copy.values(), velocities_copy)]
                    for key, param in zip(self.named_weights.keys(), current_params):
                        named_weights_copy[key] = param

                    preds_perturbed = self.model.forward(
                        adj, feature_matrix, train_dropout=False, param_dict=named_weights_copy)

                    meta_loss = (1-gamma) * metrics.cross_entropy_loss(
                        preds_perturbed[val_ids], labels[val_ids]) + gamma * metrics.cross_entropy_loss(
                        preds_perturbed[val_ids], F.softmax(preds_perturbed[val_ids], dim=1))
                    zero_grad(self.model, param_dict=named_weights_copy)
                    meta_grad_iter = torch.autograd.grad(
                        meta_loss, self.label_changes, retain_graph=True)[0]
                    meta_grad += meta_grad_iter

                    meta_loss.detach()
                    preds.detach()
                    loss.detach()
                    for key, value in named_weights_copy.items():
                        val = value.detach()
                        val.requires_grad = True
                        named_weights_copy[key] = val

            meta_grad[:, label_i] = meta_grad[:, label_i]/n_samples

            max_pred_ind = torch.argmax(pred_probs, dim=1)
            # TODO make this step more efficient
            for i in range(meta_grad.shape[0]):
                meta_grad[i, max_pred_ind[i]] = 0.
            # Calculate the expected meta-gradeint
            if self.self_supervised:
                meta_grad *= -1
            node_meta_grad = torch.sum(pred_probs*meta_grad, dim=1)

        # ignore the current prediction probabilities
        # node_meta_grad = torch.sum(meta_grad, dim=1)

        return node_meta_grad

    def _calculate_node_feature_meta_grad(self, adj, feature_matrix,
                                          labels, pred_probs, train_ids,
                                          unlabeled_ids,
                                          meta_iter=10):
        """ 
        output: node_meta_grad (self.n_nodes)
        Returns the inner product of the expected value of the gradient with
        respect to the feature matrix. We consider all possible
        self.n_classes labels for computing loss. Expected value of the inner
        product of gradient for each node is calculated using @param pred_probs.
        """
        feats_modified = feature_matrix+self.feat_changes
        preds_perturbed = self.model.forward(
            adj, feats_modified, train_dropout=False, param_dict=self.named_weights)

        # calculate the expected meta-loss
        node_meta_grads = torch.zeros((self.n_nodes, self.n_classes),
                                      device=self.device)
        for label_i in range(self.n_classes):
            labels_unseen = label_i * torch.ones(unlabeled_ids.shape[0],
                                                 device=self.device, dtype=torch.long)
            meta_loss = F.cross_entropy(
                preds_perturbed[unlabeled_ids], labels_unseen)
            self.model.zero_grad()
            meta_grad = torch.autograd.grad(
                meta_loss, self.feat_changes, retain_graph=True)
            node_meta_grads[:, label_i] = torch.pow(meta_grad[0], 2).sum(dim=1)
        meta_loss.detach()

        node_meta_grad = (node_meta_grads * pred_probs).sum(dim=1)

        return node_meta_grad


class EntropyActiveLearner(ActiveLearner):

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids):
        raise NotImplementedError


class EGLActiveLearner(ActiveLearner):

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids):
        n_unlabeled = len(unlabeled_ids)
        grad_length = torch.zeros(
            (self.n_nodes, self.n_classes), device=self.device)
        preds, graph_feats = self.model(adj, feature_matrix, mc_dropout=False,
                                        param_dict=self.named_weights, return_output=True)
        pred_probs = torch.softmax(preds, dim=1)
        for i in range(self.n_classes):
            expected_label = i * \
                torch.ones(n_unlabeled, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(
                preds[unlabeled_ids], expected_label, reduction='none')
            grads = torch.autograd.grad(loss, graph_feats, grad_outputs=torch.softmax(loss, dim=0),
                                        retain_graph=True)
            # grad_length[:, i] = torch.stack([self._calculate_grad_length(
            #     torch.autograd.grad(loss_k,
            #                         self.named_weights.values(),
            #                         retain_graph=True)) for loss_k in loss])
            grad_length[:, i] = self._calculate_grad_length(grads)
            # grad_length[i] = self._calculate_grad_length(grads)
            loss.detach()
        unlabeled_mask = torch.zeros(self.n_nodes, device=self.device)
        unlabeled_mask[unlabeled_ids] = 1
        egl = torch.sum(grad_length * pred_probs, dim=1) * unlabeled_mask
        query_ids = torch.argsort(egl, descending=True)
        query_ids = query_ids[:self.batch_size].cpu().numpy()

        return query_ids

    def _calculate_grad_length(self, gradients):
        grad_length = 0.

        for grad in gradients:
            grad_length += torch.norm(grad, dim=1)

        return grad_length


class FIRLearner(ActiveLearner):
    """
    Fisher Information Ratio-based AL model.
    ** Not complete yet **
    """

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids):
        n_unlabeled = len(unlabeled_ids)
        grad_sum = 0.
        preds, graph_feats = self.model(adj, feature_matrix, mc_dropout=False,
                                        param_dict=self.named_weights, return_output=True)
        pred_probs = torch.softmax(preds, dim=1)
        node_weights = nn.Parameter(torch.ones(n_unlabeled))

        node_weights_norm = torch.softmax(node_weights, dim=0)
        # TODO pass this as an arg
        n_inner_iters = 20
        # optimizer =

        for iter in range(n_inner_iters):
            for label_i in range(self.n_classes):
                expected_label = label_i * \
                    torch.ones(n_unlabeled, dtype=torch.long,
                               device=self.device)
                loss = F.cross_entropy(
                    preds[unlabeled_ids], expected_label, reduction='none')
                node_weights_label = node_weights * pred_probs[:, label_i]
                grads = torch.autograd.grad(loss, graph_feats,
                                            grad_outputs=node_weights_label,
                                            retain_graph=True)
                grad_sum += grads[0].norm()
                loss.detach()

            # TODO optimize node_weights
        unlabeled_mask = torch.zeros(self.n_nodes, device=self.device)
        unlabeled_mask[unlabeled_ids] = 1
        egl = torch.sum(grad_sum * pred_probs, dim=1) * unlabeled_mask
        query_ids = torch.argsort(egl, descending=True)
        query_ids = query_ids[:self.batch_size].cpu().numpy()

        return query_ids

    def _calculate_grad_length(self, gradients):
        grad_length = 0.

        for grad in gradients:
            grad_length += torch.norm(grad, dim=1)

        return grad_length
