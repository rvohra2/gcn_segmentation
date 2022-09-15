import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geometric_nn

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
        x, edge_index = data.x.cpu(), data.edge_index.cpu()
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
#       x, edge_index = data.x.cpu(), data.edge_index.cpu()
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