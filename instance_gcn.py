import torch
from torch.autograd import Variable
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import cairosvg
from PIL import Image
import io
import xml.etree.ElementTree as et
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from pathlib import Path
import networkx as nx
from convert_svg import render_svg
from dataset import BaseDataset, ChineseDataset
from gcn_model import GCNs

from utils import segmentation_adjacency, create_edge_list, create_features, create_target
from slic_segm import slic_fixed
from train import train
from test import test

###Working better for 500 epoch
Epochs = 50

edge_list = []
features_list = []

targets_list = []
datasets = []
segmentation_list = []
image_list = []
ids = BaseDataset()
for idx in range(len(ids)):
    image, masks, num_paths = ChineseDataset(ids, idx)
    image = np.asarray(image)
    image_list.append(image)

    segmentation_algorithm = slic_fixed(300, compactness=5, max_iterations=10, sigma=0)
    segmentation = segmentation_algorithm(image)
    segmentation_list.append(segmentation)


    adj = segmentation_adjacency(segmentation)
    adj = np.array(adj.todense())
    adj = torch.from_numpy(adj).type(torch.FloatTensor)

    edge_x = Variable(torch.from_numpy(create_edge_list(adj))).cuda()
    edge_list.append(edge_x)
    features = Variable(torch.from_numpy(create_features(image, segmentation)))
    features_list.append(features)


    target_mask = masks
    targets = []
    #target_mask = np.asarray(target_mask.resize((200,200)))[:, :, 1]
    for i in range(num_paths):
        y = Variable(torch.from_numpy(create_target(segmentation, target_mask, num_paths)))
        #targets.append(y)
    num_instance = np.max(target_mask)+1
    targets_list.append(y)

for i, features in enumerate(features_list):
    new_features = np.zeros((features.shape[0], 500))
    new_features[:, :features.shape[1]] = features
    new_features = Variable(torch.from_numpy(new_features)).type(torch.FloatTensor) 
    
    datasets.append(Data(new_features, edge_list[i], y=targets_list[i], segmentation = segmentation_list[i], image = image_list[i]))

loader = DataLoader(datasets, batch_size=1)

###Working better for nhid=1024, dropout=0.5, lr=0.01
model = GCNs(loader.dataset[0].x.shape[1], 1024, 2)
optimizer = optim.Adam(model.parameters(), 0.001)
all_logits = train(model, optimizer, loader, Epochs)

output_dir = Path("/home/rhythm/notebook/vectorData_test/temp/")
output_dir.mkdir(exist_ok=True, parents=True)
mask = test(model, loader, output_dir, ids, idx, 2)

# adj = segmentation_adjacency(segmentation)
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

# img, mask1 = create_mask(image, segmentation, node_num,  Epochs-1, pos, all_logits)

# #Image.blend(Image.fromarray(img), Image.fromarray(mask1), 0.5)
# #plt.figure()
# # plt.subplot(1,3,1)
# # plt.imshow(image)
# # plt.subplot(1,3,2)
# # plt.imshow(mask1)
# # plt.subplot(1,3,3)
# # plt.imshow(target_mask)
# # plt.show()

# mask1 = mask1.mean(axis=2)
# mask1 = (mask1 != 0)
# masks.append(mask1)
# del segmentation
# del mask1
# from skimage import color, segmentation
# ###Change filename number
# print(os.path.join("/home/rhythm/notebook/vectorData_test/temp/", str(idx)+ "_train.svg"), idx)
# im = render_svg(masks, os.path.join("/home/rhythm/notebook/vectorData_test/temp/", str(idx)+ "_train.svg"))




    


    

    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.show()


    ###Uncomment for line drawing dataset
    # num_paths = len(et.fromstring(svg)[0])
    # for i in range(num_paths):
    #     svg_xml = et.fromstring(svg)
    #     # svg_xml[0][0] = svg_xml[0][i]
    #     # del svg_xml[0][1:]
    #     svg_one = et.tostring(svg_xml, method='xml')

    #     # leave only one path
    #     y_png = cairosvg.svg2png(bytestring=svg_one)
    #     y_img = Image.open(io.BytesIO(y_png))
    #     mask = (np.array(y_img)[:, :, 3] > 0)
    #     mask = mask.astype(np.uint8)

    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.show()

        
    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    ###Need to fine tune
    

    #seg_img = mark_boundaries(image, segmentation)
    # fig = plt.figure("Superpixels -- %d segments" % (100))
    # plt.imshow(seg_img)
    # plt.show()

    

    

    #return adj, image, segmentation, target_mask

#loader, adj, image, segmentation, target_mask = test_loader(max_dim=-1)

    

    



    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(image)
    # plt.subplot(1,3,2)
    # plt.imshow(mask)
    # plt.subplot(1,3,3)
    # plt.imshow(target_mask)
    # plt.show()

    # mask = mask.mean(axis=2)
    # mask = (mask != 0)
    # ###Change filename number
    # im = render_svg(image, mask, node_num,  "/home/rhythm/notebook/vectorData_test/temp/10_test.svg")


    




