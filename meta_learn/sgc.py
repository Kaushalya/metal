import torch
import torch.nn as nn
import torch.nn.functional as F

class SGC(nn.Module):
    """
    Implementation of "Simplifying Graph Convolutional Networks" [Wu et al., 2019] 
    """

    def __init__(self, n_feat, n_classes, device, dropout_p):
        super(SGC, self).__init__()
        self.device = device
        self.dropout_p = dropout_p
        self.W = nn.Parameter(torch.zeros(size=(n_classes, n_feat)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.layer_names = ['W']

    def forward(self, adj, x, mc_dropout=False, dropout_p=-1., train_dropout=True, param_dict=None):
        x = F.dropout(x, p=dropout_p if dropout_p >= 0 else self.dropout_p, training=mc_dropout or (
            train_dropout and self.training))
        x = F.linear(x, param_dict['W'] if param_dict is not None
                        else self.W)
        return x


def sgc_precompute(x, adj, degree, concat=False):
    for i in range(degree):
        x = torch.sparse.mm(adj, x)

    return x
