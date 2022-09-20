import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from utils import segmentation_adjacency, create_mask

def train(model, optimizer, loader, Epochs):
    all_logits = []
    for epoch in range(Epochs):
        loss = 0
        for data in loader:
            #print("data = {}".format(data))
            y = data.y
            for i in range(len(y)):
                y_s = y[i].type(torch.cuda.LongTensor)
                
                # x = data.x
                # x = x.cpu()

                # optimizer.zero_grad()
                # output = model(x, adj).cuda()
                # all_logits.append(output.detach())
                # #output = output.transpose(0, 1)
                # loss = F.cross_entropy(output, y)
                # loss.backward()
                # optimizer.step()
                
                logits = model(data).cuda()
                all_logits.append(logits.detach())
                logp = F.log_softmax(logits, 1).cuda()
                
                loss = F.nll_loss(logp, y_s)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
                
    # adj = segmentation_adjacency(data.segmentation[0])
    # dense_adj = np.array(adj.todense())
    # edges = []
    # nodes = np.array([])
    # for i in range(0, dense_adj.shape[0]):
    #     nodes = np.append(nodes, str(i))
    # nx_G = nx.Graph()
    # nx_G.add_nodes_from(nodes)
    # nx_G.add_edges_from(edges)
    # pos = nx.spring_layout(nx_G)

    # node_num = dense_adj.shape[0]

    # img, mask = create_mask(data.image[0], data.segmentation[0], node_num,  Epochs-1, pos, all_logits)
    # plt.imshow(mask)
    # plt.show()          

            # out = model(data).cuda()  # Perform a single forward pass.
            # all_logits.append(out.detach())
            # optimizer.zero_grad()
            # loss = criterion(out, y)  # Compute the loss.
            # loss.backward()  # Derive gradients.
            # optimizer.step()  # Update parameters based on gradients.
            

        
  
    return all_logits