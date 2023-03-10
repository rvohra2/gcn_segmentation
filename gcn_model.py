import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math

import torch.nn.functional as F
import torch_geometric.nn as geometric_nn


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         #return F.log_softmax(x, dim=1)
#         return x

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)

#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




# class GraphConvolution(Module):
#         def __init__(self, in_features, out_features, batch_size=1, bias=True):
#             super(GraphConvolution, self).__init__()
            
#             self.in_features = in_features
#             self.out_features = out_features
            
#             self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#             if bias:
#                 self.bias = Parameter(torch.FloatTensor(out_features))
#             else:
#                 self.register_parameter('bias', None)
#             self.reset_parameters()
    
#         def reset_parameters(self):
#             stdv = 1. / math.sqrt(self.weight.size(1))
#             self.weight.data.uniform_(-stdv, stdv)

#             if self.bias is not None:
#                 self.bias.data.uniform_(-stdv, stdv)
    
#         def forward(self, input, adj):
            
#             support = torch.mm(input, self.weight)
#             output = torch.mm(adj, support)
#             if self.bias is not None:
#                 return output + self.bias
#             else:
#                 return output
    
#         def __repr__(self):
#             return self.__class__.__name__ + ' (' \
#                    + str(self.in_features) + ' -> ' \
#                    + str(self.out_features) + ')'
                   
# class GCNs(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCNs, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=True)
#         x = self.gc2(x, adj)
#         return x
#         #return F.log_softmax(x, dim=1)


class GCNs(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(GCNs, self).__init__()
        
        self.gcn1 = geometric_nn.GraphConv(D_in, H)
        self.gcn2 = geometric_nn.GraphConv(H, D_out)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        t = torch.relu(self.gcn1(x, edge_index))
        t = self.gcn2(t, edge_index)
        return t

# class GCNs(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         super(GCNs, self).__init__()
#         # We inherit from pytorch geometric's GCN class, and we initialize three layers
#         self.conv1 = geometric_nn.GCNConv(D_in, H)
#         self.conv2 = geometric_nn.GCNConv(H, H)
#         self.conv3 = geometric_nn.GCNConv(H, H)
#         # Our final linear layer will define our output
#         self.lin = nn.Linear(H, D_out)
        
#     def forward(self, data):
#       # 1. Obtain node embeddings 
#       x, edge_index = data.x, data.edge_index
#       x = self.conv1(x, edge_index)
#       x = x.relu()
#       x = self.conv2(x, edge_index)
#       x = x.relu()
#       x = self.conv3(x, edge_index)   
 
#       # 2. Readout layer
#       #x = geometric_nn.global_mean_pool(x)  # [batch_size, hidden_channels]
 
#       # 3. Apply a final classifier
#       x = F.dropout(x, p=0.5, training=self.training)
#       x = self.lin(x)
#       return x