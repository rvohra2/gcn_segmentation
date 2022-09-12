import torch.optim as optim
from torch_geometric.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import segmentation
import networkx as nx
from convert_svg import render_svg
from gcn_model import GCNs
from dataset import ChineseDataset
from train import train
from test import test
from utils import segmentation_adjacency

###Working better for 500 epoch
Epochs = 200

def select_mask_color_test(cls, colors):
    background_color = [0, 0, 0]
    if cls == 0:
        return background_color
    else:
        return colors[cls]

def map_to_segmentation(pred, segmentation, img_size, batch_size=1):
#    y_pred = Variable(torch.zeros(img_size)).cuda()
    y_pred = np.zeros((batch_size, img_size[0], img_size[1]))
    for i in range(batch_size):
        #print("len = {}".format(len(pred[i])))
        for j in range(len(pred[i])):
            indice = np.where(segmentation == j)
            #print("pred = {}".format(pred[i, j]))
            y_pred[i, indice[0], indice[1]] = pred[i, j]
    return y_pred

def select_mask_color(cls):
    background_color = [0, 0, 0]
    instance1_color = [255, 0, 0]
    instance2_color = [266, 251, 0]
    instance3_color = [143, 195, 31]
    instance4_color = [0, 160, 233]
    instance5_color = [29, 32, 136]
    instance6_color = [146, 7, 131]
    instance7_color = [228, 0, 79]
    
    if cls == 0:
        return background_color
    elif cls == 1:
        return instance1_color
    elif cls == 2:
        return instance2_color
    elif cls == 3:
        return instance3_color
    elif cls == 4:
        return instance4_color
    elif cls == 5:
        return instance5_color
    elif cls == 6:
        return instance6_color
    else:
        return instance7_color

def create_mask(img, segmentation, node_num, epoch):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    for v in range(0, node_num):
        pos[v] = all_logits[epoch][v].cpu().numpy()
        
        cls = pos[v].argmax()
        
        mask_color = select_mask_color(cls)
        #print(mask_color)
        mask[segmentation == v] = mask_color
        
    return img ,mask

data_dir = Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/ch/")
print(os.listdir(os.path.join(data_dir, 'train/')))
idx = len([name for name in os.listdir(os.path.join(data_dir, 'train/')) ])

for i in range(idx):
    loader, adj, image, segmentation, target_mask = ChineseDataset(data_dir, i)
    ###Working better for nhid=1024, dropout=0.5, lr=0.01
    nums, mass = np.unique(segmentation, return_counts=True)
    n = nums.shape[0]
    model = GCNs(nfeat=loader.dataset[0].x.shape[1],nhid=1024,nclass=n,dropout=0.5)
    optimizer = optim.Adam(model.parameters(), 0.01)
    all_logits = train(model, optimizer, loader, adj, Epochs)


    adj = segmentation_adjacency(segmentation)
    dense_adj = np.array(adj.todense())
    edges = []
    nodes = np.array([])
    for i in range(0, dense_adj.shape[0]):
        nodes = np.append(nodes, str(i))
    nx_G = nx.Graph()
    nx_G.add_nodes_from(nodes)
    nx_G.add_edges_from(edges)
    pos = nx.spring_layout(nx_G)

    node_num = dense_adj.shape[0]

    img, mask = create_mask(image, segmentation, node_num,  Epochs-1)

    Image.blend(Image.fromarray(img), Image.fromarray(mask), 0.5)
    plt.imshow(mask)
    plt.show()

    mask = mask.mean(axis=2)
    mask = (mask != 0)
    ###Change filename number
    im = render_svg(image, mask, node_num, "/home/rhythm/notebook/vectorData_test/temp/1_train.svg")

    # del segmentation
    # from skimage import color, segmentation
    # image, mask, target_mask = test(model, adj, n, max_dim=-1)

    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(image)
    # plt.title('Original Image')
    # plt.subplot(1,3,2)
    # plt.imshow(mask)
    # plt.title('Output Mask')
    # plt.subplot(1,3,3)
    # plt.imshow(target_mask, cmap='gray')
    # plt.title('Target Mask')
    # plt.show()

    # mask = mask.mean(axis=2)
    # mask = (mask != 0)
    # ###Change filename number
    # im = render_svg(image, mask, node_num,  "/home/rhythm/notebook/vectorData_test/temp/10_test.svg")


    




