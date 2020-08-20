from meta_learn import active_learner, metrics, active_graph_embedding
from data import data_utils
import numpy as np
from meta_learn.gcn import MetaGCN
from meta_learn.sgc import SGC, sgc_precompute
import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy
import csv
import os

from meta_learn.utils import GraphOracle
from argparse import ArgumentParser
from distutils.util import strtobool
from timebudget import timebudget
import logging

logging.basicConfig(level=logging.INFO)

def get_argparser():
    parser = ArgumentParser('Active Meta Graph Learning')
    parser.add_argument('--data_file', type=str, default='data/citeseer.npz')
    parser.add_argument('--use_sparse', type=strtobool, default='yes',
                        help='Whether to use sparse representations')
    parser.add_argument('--normalize_adj', type=strtobool, default='yes',
                        help='Whether to normalize the adjacency matrix' +
                        ' when performing graph convolutions')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of nodes queried in a single step')
    parser.add_argument('--query_ratio', type=float, default=0.1,
                        help='Number of times to be queried')
    parser.add_argument('--n_queries', type=int, default=10,
                        help='Number of times to be queried')
    parser.add_argument('--init_size', type=float, default=0.1,
                        help='size of the initial random sample of labeled nodes')
    parser.add_argument('--n_train_per_class', type=int, default=2,
                        help='number of items to be selected to as the intial labeled set')
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'sgc'])
    parser.add_argument('--model_train_epoch', type=int, default=50)
    parser.add_argument('--active_train_epoch', type=int, default=20)
    parser.add_argument('--mc_iter', type=int, default=10)
    parser.add_argument('--meta_iter', type=int, default=5)
    parser.add_argument('--debug', type=strtobool, default='no')
    parser.add_argument('--seed', type=int, default=1023)
    parser.add_argument('--gpu', type=strtobool, default='yes')
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='wieght decay (L2 loss on parameters)')
    parser.add_argument('--policy', type=str, default='meta',
                        choices=['meta', 'random', 'entropy', 'both', 'egl',
                                 'age', 'pagerank', 'bald', 'degree'])
    parser.add_argument('--dropout_p', type=float, default='0.5',
                        help='Dropout rate = (1 - keep-probability)')
    parser.add_argument('--mc_dropout', type=strtobool, default='yes')
    parser.add_argument('--balanced_split', type=strtobool, default='yes',
                        help='An equal number of samples are chosen as the intial sample')
    parser.add_argument('--mc_dropout_iter', type=int, default=10,
                        help='Number of iterations of MC-dropout. Used for\
                             calculating model uncertainty')
    parser.add_argument('--n_experiments', type=int, default=1)
    parser.add_argument('--approx_meta_grad', type=strtobool, default='no',
                        help='Whether to approximate meta gradients instead of calculating' +
                        'second order derivatives')
    parser.add_argument('--self_supervised', default=False, action='store_true',
                        help='uses labels infered by the model when calculating meta-gradient')
    parser.add_argument('--query_strategy', default='greedy',
                        choices=['greedy', 'eps_greedy', 'ucb'])
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--ucb_beta', type=float, default=0.5,
                        help='Exploration coefficient beta used in UCB-style algorithm')
    parser.add_argument('--save_logs', default=False, action='store_true',
                        help='Save the node ids acquired in each iteration to a text file')
    args = parser.parse_args()
    return args


def run_active_learner(args, n_nodes, train_share, test_share, unlabeled_share, adj_normalized, x,
                       true_labels_1d, true_labels_2d, device, random_state=None, results_file_name=None):
    if args.balanced_split:
        split_train, split_test, split_unlabeled = data_utils.train_test_split_balanced(np.arange(n_nodes), labels=true_labels_1d.cpu().numpy(),
                                                                                        n_train_per_class=args.n_train_per_class,
                                                                                        test_size=test_share,
                                                                                        unlabeled_size=unlabeled_share,
                                                                                        random_state=random_state)
    else:
        split_train, split_test, split_unlabeled = data_utils.train_val_test_split_tabular(np.arange(n_nodes),
                                                                                          train_size=train_share,
                                                                                          val_size=test_share,
                                                                                          test_size=unlabeled_share,
                                                                                          stratify=_z_obs,
                                                                                          random_state=random_state)
    train_dropout = True
    if args.model == 'gcn':
        gnn_model = MetaGCN(x.shape[1], true_labels_2d.shape[1], device, dropout_p=args.dropout_p,
                    sparse=args.use_sparse, ch_list=[x.shape[1], 64])
    elif args.model == 'sgc':
        train_dropout = False
        x = sgc_precompute(x, adj_normalized, degree=3)
        gnn_model = SGC(x.shape[1], true_labels_2d.shape[1], device, dropout_p=args.dropout_p)

    if args.gpu and torch.cuda.is_available():
        gnn_model = gnn_model.to(device)

    optimizer = optim.Adam(gnn_model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    # train the model with initial training set
    gnn_model.train()
    target = true_labels_1d
    loss_func = F.cross_entropy

    for iter in range(args.model_train_epoch):
        optimizer.zero_grad()
        preds = gnn_model(adj_normalized, x, train_dropout=train_dropout)
        loss_train = loss_func(
            preds[split_train], target[split_train])
        loss_train.backward()
        optimizer.step()
        acc_train = metrics.accuracy(
            preds[split_train], target[split_train])
        acc_val = metrics.accuracy(preds[split_test], target[split_test])
        logging.info("epoch={}, train-loss={:.3f}, train-accuracy={:.2f}, val-accuracy={:.2f}".
              format(iter, loss_train.item(), acc_train.item(), acc_val.item()))

    logging.info("Model training is complete.")
    # evaluation mode
    gnn_model.eval()
    preds = gnn_model(adj_normalized, x)
    loss_train = F.cross_entropy(
        preds[split_train], true_labels_1d[split_train])
    acc_train = metrics.accuracy(
        preds[split_train], true_labels_1d[split_train])
    acc_val = metrics.accuracy(preds[split_test], true_labels_1d[split_test])
    acc_test = metrics.accuracy(
        preds[split_unlabeled], true_labels_1d[split_unlabeled])
    logging.info("train-loss={:.3f}, train-accuracy={:.2f}, \
     val-accuracy={:.2f} test-accuracy={:.2f}".format(loss_train.item(),
                                                      acc_train.item(),
                                                      acc_val.item(), acc_test.item()))
    if results_file_name:
        with open(results_file_name, 'a') as results_file:
            results_file.write(f'initial sample: {split_train.tolist()}\n')
            results_file.write(f'train accuracy = {acc_train.item():.3f}\n')
            results_file.write(f'test accuracy = {acc_test.item():.3f}\n')
    
    gnn_model.train()
    # graph_oracle = GraphOracle(true_labels_1d)
    graph_oracle = GraphOracle(true_labels_2d)

    observed_labels_2d = torch.zeros_like(true_labels_2d)
    observed_labels_2d[split_train, :] = true_labels_2d[split_train, :]
    observed_labels_2d[split_test, :] = true_labels_2d[split_test, :]
    observed_labels_2d[split_unlabeled, :] = torch.softmax(preds[split_unlabeled, :],
                                                           dim=1)

    # observed_labels_2d = torch.softmax(observed_labels_2d, dim=1)
    observed_labels_2d = observed_labels_2d.detach()
    observed_labels_1d = observed_labels_2d.argmax(1)

    if args.policy == 'random':
        meta_learner = active_learner.RandomActiveLearner(gnn_model, adj.shape[0],
                                                        true_labels_2d.shape[1],
                                                        x.shape[1],
                                                        graph_oracle,
                                                        device=device,
                                                        train_steps=args.active_train_epoch,
                                                        learning_rate=args.learning_rate,
                                                        output_file_name=results_file_name,
                                                        batch_size=args.batch_size,
                                                        policy=args.policy,
                                                        optimizer=optimizer)
    elif args.policy in ('age', 'pagerank', 'degree'):
        meta_learner = active_graph_embedding.CentralityActiveLearner(gnn_model, adj.shape[0],
                                                                      true_labels_2d.shape[1],
                                                                      x.shape[1],
                                                                      graph_oracle,
                                                                      device=device,
                                                                      train_steps=args.active_train_epoch,
                                                                      learning_rate=args.learning_rate,
                                                                      debug=args.debug,
                                                                      output_file_name=results_file_name,
                                                                      batch_size=args.batch_size,
                                                                      policy=args.policy,
                                                                      query_strategy=args.query_strategy,
                                                                      optimizer=optimizer)
    elif args.policy == 'bald':
        meta_learner = active_learner.BALDActiveLearner(gnn_model, adj.shape[0],
                                                        true_labels_2d.shape[1],
                                                        x.shape[1],
                                                        graph_oracle,
                                                        device=device,
                                                        train_steps=args.active_train_epoch,
                                                        learning_rate=args.learning_rate,
                                                        output_file_name=results_file_name,
                                                        batch_size=args.batch_size,
                                                        policy=args.policy,
                                                        query_strategy=args.query_strategy,
                                                        ucb_beta=args.ucb_beta,
                                                        epsilon=args.epsilon,
                                                        mc_dropout=args.mc_dropout,
                                                        mc_dropout_iter=args.mc_dropout_iter,
                                                        use_infod=args.use_infod,
                                                        optimizer=optimizer)
    else:
        meta_learner = active_learner.MetaActiveLearner(gnn_model, adj.shape[0],
                                                        true_labels_2d.shape[1],
                                                        x.shape[1],
                                                        graph_oracle,
                                                        device=device,
                                                        train_steps=args.active_train_epoch,
                                                        learning_rate=args.learning_rate,
                                                        batch_size=args.batch_size,
                                                        policy=args.policy,
                                                        query_strategy=args.query_strategy,
                                                        ucb_beta = args.ucb_beta,
                                                        epsilon=args.epsilon,
                                                        output_file_name=results_file_name,
                                                        mc_dropout=args.mc_dropout,
                                                        mc_dropout_iter=args.mc_dropout_iter,
                                                        optimizer=optimizer
                                                        )

    if args.gpu and torch.cuda.is_available():
        meta_learner.cuda()

    timebudget.set_quiet()
    timebudget.report_at_exit()

    with timebudget("Active Learner acquiring data"):
        train_ids, unlabeled_ids, performance_dict = meta_learner(adj_normalized, x, observed_labels_2d, split_train, split_test,
                                                                  split_unlabeled, n_queries=args.n_queries,
                                                                  return_performance=True)
    logging.debug("Final query ids: ", np.setdiff1d(
        train_ids, split_train).tolist())
    if results_file_name:
        with open(results_file_name, 'a') as results_file:
            results_file.write(f'acquisition: {train_ids.tolist()}\n')

    return performance_dict


if __name__ == "__main__":
    args = get_argparser()
    if args.data_file.endswith('.npz'):
        adj, _x, _z_obs = data_utils.load_npz(args.data_file)
    elif args.data_file.endswith('pubmed'):
        adj, _x, _z_obs = data_utils.load_data(args.data_file, 'pubmed')
    else:
        adj, _x, _z_obs = data_utils.load_ppi(args.data_file)
    adj = adj + adj.T
    adj[adj > 1] = 1
    lcc = data_utils.largest_connected_components(adj)

    adj = adj[lcc][:, lcc]
    adj.setdiag(0)
    adj = adj.astype("float32")
    adj.eliminate_zeros()
    _x = _x.astype("float32")

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(
        np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
    # TODO uncomment if needed?
    assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    _x = _x[lcc]
    _z_obs = _z_obs[lcc]
    n_nodes = adj.shape[0]
    n_classes = _z_obs.max()+1
    _Z_obs = np.eye(n_classes)[_z_obs]
    degrees = adj.sum(0).A1

    # unlabeled_share = 0.895
    test_share = 0.05
    # train_share = 1 - unlabeled_share - val_share
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.balanced_split:
        train_share = (args.n_train_per_class * n_classes)/n_nodes
        unlabeled_share = 1 - train_share - test_share
    else:
        train_share = int(args.init_size * n_nodes)
        test_share = int(test_share * n_nodes)
        unlabeled_share = n_nodes - train_share - test_share

    # split_unlabeled = np.union1d(split_val, split_unlabeled)
    # n_queries = int(args.query_ratio * n_nodes)
    # dtype = np.float32
    device = torch.device('cpu')

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Running on GPU : {}".format(device))

    if scipy.sparse.issparse(adj):
        adj = data_utils.preprocess_sparse_adj(adj)
        adj = data_utils.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = torch.FloatTensor(adj)
        adj = data_utils.preprocess_adj(adj, device=device)
    
    if scipy.sparse.issparse(_x):
        _x = _x.todense()
    x = torch.FloatTensor(_x)
    true_labels_1d = torch.LongTensor(_z_obs.astype(np.int))
    true_labels_2d = torch.FloatTensor(_Z_obs.astype(np.int))

    if args.gpu and torch.cuda.is_available():
        adj = adj.to(device)
        x = x.to(device)
        true_labels_1d = true_labels_1d.to(device)
        true_labels_2d = true_labels_2d.to(device)

    test_acc = list()

    # Create a file to save the output of each run of the acquisition algorithm
    dataset_name = os.path.basename(args.data_file).split('.')[0]
    log_file_name = None
    policy = args.policy
    if args.query_strategy == 'ucb':
        if args.ucb_beta > 0.:
            policy = f'{args.policy}_ucb0{int(args.ucb_beta*10)}'
        elif args.ucb_beta < 0.:
            policy = f'{args.policy}_ucb'
    elif args.query_strategy == 'eps_greedy':
        policy = f'{args.policy}_eps0{int(args.epsilon*10)}'

    if args.save_logs:
        os.makedirs('results/logs', exist_ok=True)
        log_file_name = f'results/logs/{dataset_name}_{policy}_batch{args.batch_size}.txt'

        with open(log_file_name, 'w') as results_file:
            results_file.write(f'dataset: {args.data_file}\n')
            results_file.write(f'batch size: {args.batch_size}\n')
            results_file.write(f'learning rate: {args.learning_rate}\n')
            results_file.write(f'policy: {args.policy}\n')

    for i in range(args.n_experiments):
        rand_state = np.random.RandomState(i+100)
        torch.manual_seed(i+100)
        logging.info("running experiment: {}".format(i))
        if args.save_logs:
            with open(log_file_name, 'a') as results_file:
                results_file.write(f'Experiment: {i}\n')

        performance = run_active_learner(args, n_nodes, train_share, test_share,
                                         unlabeled_share, adj, x, true_labels_1d, true_labels_2d, device,
                                         random_state=rand_state, results_file_name=log_file_name)
        test_acc.append(performance['test-accuracy'])
        test_acc.append(performance['macro-f1'])
        test_acc.append(performance['micro-f1'])

    test_acc = np.transpose(np.array(test_acc))
    np.savetxt(f'results/{dataset_name}_{policy}_{args.n_experiments}trials-accuracy.csv',
               test_acc, delimiter=",")
