import csv
import logging
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch import distributions
from torch.utils.data.sampler import WeightedRandomSampler
from meta_learn import metrics
from .utils import avg_shortest_path_length, torch_to_graph, zero_grad, entropy,\
    logit_mean, mutual_information, get_model_weights, sgc_precompute
from .utils import calculate_info_density, perc, percd
import logging

from collections import defaultdict
from copy import deepcopy


def calculate_mc_metagrad(learner: 'MetaActiveLearner', adj, feature_matrix,
                          labels, pred_probs, train_mask, unlabeled_mask,
                          epoch, query_ids=None, val_ids=None,
                          alpha=0.2, beta=1.0,
                          gamma=0.2, meta_iter=10,
                          val_share=0.2, dropout_p=0.):
    # Randomly select a set of unlabeled items (unless given) as validation.
    # Add perturbations to the labels of the remaining unlabeled items and
    # retrain the model.
    labels_modified = torch.softmax(labels*(1+learner.label_changes), dim=1)
    meta_grad = torch.zeros_like(learner.label_changes, device=learner.device)
    labels_1d = labels.argmax(1)
    val_given = False
    entropy_loss = True
    weighted_loss = False
    learner.model.train()

    if val_ids is not None:
        val_given = True
        val_mask = torch.zeros_like(unlabeled_mask)
        val_mask[val_ids] = True
        unlabeled_ids_iter = unlabeled_mask & ~val_mask

    named_weights = get_model_weights(learner.model)
    velocities = [torch.zeros_like(w) for w in named_weights.values()]

    # Randomly select a set of unlabeled nodes as the validation set.
    # This validation set is used to calculate the meta-gradient.
    if not val_given:
        k_ = int(val_share * sum(unlabeled_mask))
        val_ids = torch.where(unlabeled_mask)[0][torch.randperm(sum(unlabeled_mask))[:k_]]
        val_mask = torch.zeros_like(unlabeled_mask, device=learner.device)
        val_mask[val_ids] = True
        unlabeled_ids_iter = unlabeled_mask & ~val_mask

    for m_iter in range(meta_iter):
        preds = learner.model.forward(
            adj, feature_matrix, param_dict=named_weights,
            train_dropout=False)
        # calculate cross-entropy loss between predictions and target
        loss = metrics.cross_entropy_loss(preds[unlabeled_ids_iter],
                                            labels_modified[unlabeled_ids_iter])
        logging.debug(f"epoch: meta {m_iter} train-loss = {loss.data}")
        zero_grad(params=named_weights)
        grads = torch.autograd.grad(
            loss, [w for w in named_weights.values()], create_graph=True,
            retain_graph=True)
        velocities = [(learner.momentum * v) +
                            grad for grad, v in zip(grads, velocities)]
        # update weights using SGD with momentum
        current_params = [w - (learner.learning_rate * v) for w,
                            v in zip(named_weights.values(), velocities)]
        for key, param in zip(named_weights.keys(), current_params):
            named_weights[key] = param

    preds_perturbed = learner.model.forward(adj, feature_matrix,
                                            train_dropout=True,
                                            dropout_p=dropout_p,
                                            param_dict=named_weights)
    if entropy_loss:
        if weighted_loss:
            gamma = 0.8**epoch
            label_weights = torch.sum(labels[train_mask], dim=0)
            if query_ids is not None and any(query_ids):
                label_weights += torch.sum(labels[query_ids], dim=0)
            label_weights = torch.softmax(label_weights, dim=0)
            label_weights = torch.pow(1-label_weights, gamma)
        else:
            label_weights = None

        meta_loss = alpha * F.cross_entropy(preds_perturbed[train_mask], labels_1d[train_mask])\
            + beta * torch.mean(entropy(
                torch.log_softmax(preds_perturbed[val_ids], dim=1),
                dim=1, weights=label_weights))
    else:
        meta_loss = alpha * F.cross_entropy(preds_perturbed[train_mask], labels_1d[train_mask])\
        + beta * metrics.cross_entropy_loss(preds_perturbed[val_ids],
                                            labels[val_ids])
    zero_grad(params=learner.label_changes)
    meta_grad = torch.autograd.grad(
        meta_loss, learner.label_changes, retain_graph=True)[0]
    # n_samples[unlabeled_ids_iter] += 1
    meta_loss.detach()
    preds.detach()
    loss.detach()

    meta_grad *= -1
    max_pred_ind = torch.argmax(pred_probs, dim=1)
    for i in range(meta_grad.shape[0]):
        meta_grad[i, max_pred_ind[i]] = 0.

    return torch.sum(pred_probs*meta_grad, dim=1)

class ActiveLearner(nn.Module):
    def __init__(self, model, n_nodes, n_classes, n_features, oracle, query_strategy=None,
                 train_steps=100, learning_rate=0.1, momentum=0.9, debug=False,
                 batch_size=1,
                 ucb_beta = 0.,
                 epsilon = 0.,
                 device=None,
                 normalize_adj=True,
                 policy="meta",
                 mc_dropout=False,
                 mc_dropout_iter=10,
                 output_file_name=None,
                 optimizer=None):
        super(ActiveLearner, self).__init__()
        self.n_nodes = n_nodes
        self.n_classes = n_classes
        self.model = model
        self.oracle = oracle
        self.query_strategy = query_strategy
        self.ucb_beta = ucb_beta
        self.epsilon = epsilon
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.debug = debug
        self.batch_size = batch_size
        self.device = device
        self.normalize_adj = normalize_adj
        self.policy = policy
        self.mc_dropout = mc_dropout
        self.mc_dropout_iter = mc_dropout_iter
        self.named_weights = None
        self.output_file_name = output_file_name
        self.optimizer = optimizer
        logging.info(self)

    def forward(self, adj, feature_matrix, labels, train_ids, test_ids,
                unlabeled_ids, n_queries, return_performance=False, freq_analysis=False,
                distance_analysis=False):
        performance_dict = defaultdict(list)
        unlabeled_mask = torch.zeros(self.n_nodes, dtype=torch.bool,
                                         device=self.device)
        train_mask = torch.zeros_like(unlabeled_mask, device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        train_mask[train_ids] = True

        for i in range(n_queries):
            self.model.train()
            epsilon = 0.1
            if self.query_strategy == 'eps_greedy' and np.random.random() <= epsilon:
                query_ids = np.random.choice(
                    unlabeled_ids, self.batch_size, replace=False)
            else:
                criteria = self.select_query_nodes(adj, feature_matrix,
                                                            labels, train_mask,
                                                            unlabeled_mask, i)
                if self.query_strategy == 'ucb':
                    if self.ucb_beta == -1:
                        ucb_beta = np.random.beta(1.0, 0.2*(i+1))
                    else:
                        ucb_beta = self.ucb_beta
                    criteria = (criteria-criteria.min()) / \
                        (criteria.max()-criteria.min())
                    conf = torch.log(torch.sparse.mm(
                        adj, unlabeled_mask.float().view((-1, 1)))+1e-5).view((-1))
                    conf = (conf-conf.min())/(conf.max()-conf.min())
                    alpha = 1.-ucb_beta

                    if self.infod is not None:
                        ucb_beta /= 2
                    criteria = alpha*criteria + ucb_beta*conf

                criteria *= unlabeled_mask.float()
                _, query_ids = criteria.topk(self.batch_size, largest=True)
                query_ids = query_ids.cpu().numpy()

            train_mask[query_ids] = True
            unlabeled_mask[query_ids] = False
            # Observe the label of nodes in query_ids
            labels[query_ids] = self.oracle.get_labels(query_ids)
            labels_1d = labels.argmax(dim=1)

            if self.optimizer:
                self._train_optimizer(self.optimizer,
                                      adj, feature_matrix, labels_1d, train_mask)
            else:
                # reinitialize the model
                for k, w in self.named_weights.items():
                    nn.init.xavier_normal_(w.data, gain=1.414)
                self.train_classifier(adj, feature_matrix, labels_1d, train_mask)
            if return_performance:
                metrics = self._get_performance_metrics(
                    adj, feature_matrix, labels_1d, query_ids, train_mask, test_ids,
                    i, freq_analysis, distance_analysis)
                for key, val in metrics.items():
                    performance_dict[key].append(val)

                if self.output_file_name:
                    with open(self.output_file_name, 'a') as results_file:
                        results_file.write(f'{query_ids.tolist()}\n')
            
        if return_performance:
            return (train_ids, unlabeled_ids, performance_dict)

        return (train_ids, unlabeled_ids)

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_id, epoch, **kwargs):
        """
        Abstract function, this should be overridden by child classes with the logic 
        for selecting query nodes.
        """
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

    def _get_performance_metrics(self, adj, feature_matrix, labels, query_ids, train_mask,
                                 test_ids, iter_id, freq_analysis,
                                 distance_analysis):
        metric_dict = {}
        self.model.eval()
        with torch.no_grad():
            preds = self.model(adj, feature_matrix,
                               param_dict=self.named_weights)
            metric_dict['train-accuracy'] = metrics.accuracy(preds[train_mask],
                                                             self.oracle.get_labels(train_mask))
            metric_dict['test-accuracy'] = metrics.accuracy(
                preds[test_ids], self.oracle.get_labels(test_ids))
            metric_dict['macro-f1'] = metrics.f1_score(
                preds[test_ids], self.oracle.get_labels(test_ids), average='macro')
            metric_dict['micro-f1'] = metrics.f1_score(
                preds[test_ids], self.oracle.get_labels(test_ids), average='micro')
            logging.info("train set size: {}".format(train_mask.sum()))
            summary_str = ", ".join(["{} = {:.3f}".format(x, y)
                                     for x, y in metric_dict.items()])
            logging.info(f"perturbation:{iter_id+1} " + summary_str)

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
            zero_grad(params=self.named_weights)
            preds = self.model.forward(
                adj, feature_matrix, param_dict=self.named_weights)
            loss = F.cross_entropy(preds[train_ids], labels[train_ids])
            logging.debug(f"epoch:{iter} train-loss = {loss.data}")
            # self.model.zero_grad(param_dict=self.named_weights)
            grads = torch.autograd.grad(
                loss, self.named_weights.values(), create_graph=create_graph)
            self.velocities = [(self.momentum * v) +
                               grad for grad, v in zip(grads, self.velocities)]
            current_params = [w - (self.learning_rate * v) for w,
                              v in zip(self.named_weights.values(), self.velocities)]
            self.set_weights(current_params)

    def _train_optimizer(self, optimizer, adj, feature_matrix, labels, train_ids):
        self.model.train()
        for iter in range(self.train_steps):
            optimizer.zero_grad()
            preds = self.model(adj, feature_matrix)
            loss_train = F.cross_entropy(
                preds[train_ids], labels[train_ids])
            loss_train.backward()
            optimizer.step()

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
        return torch.rand(self.n_nodes, device=self.device)


class BALDActiveLearner(ActiveLearner):
    """
    Implementation of the paper "Deep Bayesian Active Learning with Image Data"
    by Yarin Gal et al. (2017)
    """

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids, epoch, **kwargs):
        logits = self.forward_mc(adj, feature_matrix)
        return mutual_information(logits)


class MetaActiveLearner(ActiveLearner):
    def __init__(self, model, n_nodes, n_classes, n_features, oracle, query_strategy=None,
                 train_steps=100, learning_rate=0.1, momentum=0.9, debug=False,
                 batch_size=1,
                 device=None,
                 normalize_adj=True,
                 policy="meta",
                 ucb_beta=0.,
                 epsilon=0.,
                 mc_dropout=False,
                 mc_dropout_iter=10,
                 output_file_name=None,
                 optimizer=None):
        super(MetaActiveLearner, self).__init__(model, n_nodes, n_classes,
                                                n_features, oracle, train_steps=train_steps,
                                                learning_rate=learning_rate, policy=policy,
                                                query_strategy=query_strategy,
                                                ucb_beta=ucb_beta,
                                                epsilon=epsilon,
                                                momentum=momentum, batch_size=batch_size,
                                                device=device, mc_dropout=mc_dropout,
                                                output_file_name=output_file_name,
                                                optimizer=optimizer)
        self.mc_dropout_iter = mc_dropout_iter
        self.meta_learning_rate = 1e-1
        # TODO load this from a config
        # self.meta_grad_type = 'mc'

        self.label_changes = nn.Parameter(
            torch.zeros(size=(n_nodes, n_classes), device=self.device))
        nn.init.uniform_(self.label_changes.data, 0.1, 0.5)
        self.named_weights = None

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids, unlabeled_ids, epoch,
                           **kwargs):
        unlabeled_mask = torch.zeros(self.n_nodes, dtype=torch.bool,
                                         device=self.device)
        unlabeled_mask[unlabeled_ids] = True
        meta_ucb = False

        node_entropy = torch.ones((self.n_nodes), dtype=torch.float32,
                                  device=self.device)
        node_meta_grad = torch.ones_like(node_entropy)
        train_mask = torch.zeros_like(unlabeled_mask, device=self.device)                             
        train_mask[train_ids] = True
        query_ids = None
        logits = self.forward_mc(adj, feature_matrix)
        pred_probs = torch.exp(logits).mean(dim=0)
        node_entropy = self.calculate_entropy(logits)
        alpha = 1.0

        if self.policy == 'meta' or self.policy == 'both':
            thompson_sampling = False
            val_share = 0.1
            # dropout_p = max(0.1, 0.65**(epoch+1)-0.05)
            dropout_p = 0.0

            val_ids = torch.topk(node_entropy*unlabeled_mask, int(val_share * sum(unlabeled_mask)),
                                    largest=True)[1]
            node_meta_grad = calculate_mc_metagrad(self, adj, feature_matrix, labels, pred_probs,
                                                    train_mask, unlabeled_mask, epoch,
                                                    val_ids=val_ids,
                                                    dropout_p=dropout_p)
        if self.policy == 'meta':
            criteria = node_meta_grad
        else:
            criteria = node_entropy * node_meta_grad

        return criteria
        

class EntropyActiveLearner(ActiveLearner):

    def select_query_nodes(self, adj, feature_matrix, labels, train_ids,
                           unlabeled_ids):
        raise NotImplementedError
