import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import gc

from utils import segmentation_adjacency, create_mask, save_ckp, load_ckp

def train(model, optimizer, loader, start_epoch, Epochs):
    all_logits = []
    for epoch in range(start_epoch, Epochs):
        model.train()
        loss = 0
        for data in loader:
            gc.collect()
            torch.cuda.empty_cache()
            #print("data = {}".format(data))
            y = data.y
            y_s = y.type(torch.cuda.LongTensor)
            #y_s = torch.tensor(y, dtype=int)
            
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
            logp = F.log_softmax(logits)
            #print('logp: ', logp.max(), logp.min())
            #print('y_s: ', y_s.min(), y_s.max())
            
            loss = F.nll_loss(logp, y_s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': loss.item(),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if epoch%10 == 0:
            # save checkpoint
            save_ckp(checkpoint, False, "/home/rhythm/notebook/vectorData_test/temp/chk.pt", "/home/rhythm/notebook/vectorData_test/temp/model.pt")
                
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