import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaGCN(nn.Module):
    """
    Implementation of Graph Convolutional Networks (Kipf and Welling, ICLR 2017)
    """

    def __init__(self, n_feat, n_classes, device, dropout_p=0., ch_list=None,
                 sparse=False):
        """
        -- params
        n_feat: number of input features
        n_classes: number of labels
        device: device 
        dropout_p: dropout probability
        ch_list: list of channel dimensions
        sparse: if True, sparse matrix operations are used
        """
        super(MetaGCN, self).__init__()
        n_hidden = 16
        ch_list = ch_list or [n_feat, n_hidden]
        self.device = device
        self.dropout_p = dropout_p
        self.gconvs = [GraphLinearLayer(
            ch_list[i], ch_list[i+1], device, sparse=sparse) for i in range(len(ch_list)-1)]
        self.lin = GraphLinearLayer(ch_list[-1], n_classes, device, sparse=sparse)
        self.layer_names = ['lin']
        # self.normalize_adj = normalize_adj
        self.eye = None
        for i, gconv in enumerate(self.gconvs):
            layer_name = 'gconv_{}'.format(i)
            self.add_module(layer_name, gconv)
            self.layer_names.append(layer_name)

    def forward(self, adj, x, mc_dropout=False, dropout_p=-1.,
                train_dropout=True, param_dict=None,
                return_output=False):
        """
        Forward step of the GCN with an added functionality of using weights passed as an 
        external dictionary. This is useful for inner-loop optimization in a meta-learning setting.
        :param adj: adjacency matrix
        :param x: feature matrix
        :param param_dict: a dictionary of weights to be used as model parameters
        :param return_output: if this is True, returns the output of the last graph 
        convolutional layer
        """
        for i, gconv in enumerate(self.gconvs):
            # params = None
            # if param_dict is not None:
            #     params = param_dict['gconv_{}'.format(i)]
            x = F.relu(gconv(adj, x, params=param_dict['gconv_{}'.format(
                i)] if param_dict is not None else None))
            if return_output:
                output = x
        x = F.dropout(x, p=dropout_p if dropout_p>=0 else self.dropout_p, training=mc_dropout or (
            train_dropout and self.training))
        h = self.lin(
            adj, x, params=param_dict['lin'] if param_dict is not None else None)
        if return_output:
            return h, output

        return h

    def zero_grad(self, param_dict=None):
        if param_dict is None:
            super().zero_grad()
        else:
            for name, param in param_dict.items():
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()
                    param_dict[name].grad = None


class GraphLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device, sparse=False):
        """
        Implementation of linear layer of GCN.
        Sparse operations are not supported yet.
        """
        super(GraphLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.sparse = sparse
        self.W = nn.Parameter(torch.zeros(
                size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

    def forward(self, adj, x, params=None):
        # if params is None:
        #     params = self.W
        # h: N x out
        h = torch.mm(x, params if params is not None else self.W)
        if self.sparse:
            h = torch.sparse.mm(adj, h)
        else:
            h = torch.mm(adj, h)
        return h
