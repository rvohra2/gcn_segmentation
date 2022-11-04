from json import load
import torch
import argparse
from torch.autograd import Variable
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
from dataset import BaseDataset, get_dataset, ChineseDataset, QuickDrawDataset
from gcn_model import GCNs
from utils import segmentation_adjacency, create_edge_list, create_features, create_target, load_ckp
from slic_segm import slic_fixed
from train import main
from test import test
from random import shuffle
import timeit
import os.path as osp
###Working better for 500 epoch

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1500, help='number of epochs to train our network for')
parser.add_argument('-t', '--split', type=str, default='test', help='train/val/test')
parser.add_argument('-lr', '--lr', type=float, default='1e-5', help='learning rate')
args = vars(parser.parse_args())

Epochs = args['epochs']
split = args['split']
lr = args['lr']

edge_list = []
features_list = []
targets_list = []

segmentation_list = []

#datasets = BaseDataset((Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/qdraw/cat")), split, False )
dataset = get_dataset('cat', split, False)

# # else:
# #     dataset_untrans = get_dataset('cat', split, False)
# #     dataset_trans = get_dataset('cat', split, True)
# #     dataset = torch.utils.data.ConcatDataset([dataset_untrans, dataset_trans])
# for idx in range(len(dataset)):
#     image, masks, num_paths = dataset[idx]
#     image = np.asarray(image)
    
#     #start_seg = timeit.default_timer()
#     segmentation_algorithm = slic_fixed(1000, compactness=50, max_iterations=10, sigma=0)
#     segmentation = segmentation_algorithm(image)
#     segmentation_list.append(Variable(torch.from_numpy(segmentation)))
#     #stop_seg = timeit.default_timer()
#     #print('Time Seg: ', stop_seg - start_seg)

#     #start_adj = timeit.default_timer()
#     adj = segmentation_adjacency(segmentation)
#     adj = np.array(adj.todense())
#     adj = torch.from_numpy(adj).type(torch.FloatTensor)
#     #stop_adj = timeit.default_timer()
#     #print('Time adj: ', stop_adj - start_adj)

#     #start_edge = timeit.default_timer()
#     edge_x = Variable(torch.from_numpy(create_edge_list(adj)))
#     #edge_x = create_edge_list(adj)
#     edge_list.append(edge_x)
#     #stop_edge = timeit.default_timer()
#     #print('Time edge: ', stop_edge - start_edge)

#     #start_fea = timeit.default_timer()
#     features = Variable(torch.from_numpy(create_features(image, segmentation))).type(torch.FloatTensor)

#     features_list.append(features)
#     #stop_fea = timeit.default_timer()
#     #print('Time fea: ', stop_fea - start_fea)

#     #start_y = timeit.default_timer()
#     target_mask = masks
#     #print(np.unique(target_mask))
#     targets = []
#     y = Variable(torch.from_numpy(create_target(segmentation, target_mask, num_paths)))
#     num_instance = np.max(target_mask)+1
#     targets_list.append(y)
#     data = Data(features, edge_x, y=y, segmentation = segmentation)
#     #datasets.append(data)
#     torch.save(data, osp.join("/home/rhythm/notebook/vectorData_test/temp/processed/", f'data_{idx}.pt'))
    
#     del image, features, segmentation, edge_x, y, adj
    #stop_y = timeit.default_timer()
    #print('Time y: ', stop_y - start_y)

# for i, features in enumerate(features_list):
#     new_features = np.zeros((features.shape[0], 500))
#     new_features[:, :features.shape[1]] = features
#     new_features = Variable(torch.from_numpy(new_features)).type(torch.FloatTensor) 
#     data = Data(new_features, edge_list[i], y=targets_list[i], segmentation = segmentation_list[i])
#     datasets.append(data)
#     torch.save(datasets, osp.join("/home/rhythm/notebook/vectorData_test/temp/processed.pt", ))

ckp_path = "/home/rhythm/notebook/vectorData_test/temp/chk.pt"
start_epoch = 0
valid_loss_min = np.inf
print('Pass DS')
datasets = []
for idx in range(len(dataset)):
    datasets.append(torch.load(osp.join("/home/rhythm/notebook/vectorData_test/temp/processed/", f'data_{idx}.pt')))
#datasets = [torch.load(osp.join("/home/rhythm/notebook/vectorData_test/temp/processed1/", f'data_{idx}.pt')) for idx in range(len(dataset))]

if split == 'train':
    #shuffle(datasets)
    num_sample = len(datasets)
    print(num_sample)
    split_idx = int(0.8*num_sample)
    train_ds = datasets[:split_idx]
    val_ds = datasets[split_idx:]
    shuffle(train_ds)
    shuffle(val_ds)
    loader = DataLoader(train_ds, batch_size=1)
    val_loader = DataLoader(val_ds, batch_size=1)
    model = GCNs(loader.dataset[0].x.shape[1], 1024, 30)
    optimizer = optim.Adam(model.parameters(), lr)
    all_logits = main(model, optimizer, loader, val_loader, start_epoch, Epochs, valid_loss_min)

else:
    loader = DataLoader(datasets, batch_size=1)
    model = GCNs(loader.dataset[0].x.shape[1], 1024, 30)
    optimizer = optim.Adam(model.parameters(), lr)
    model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
    model.eval()
    output_dir = Path("/home/rhythm/notebook/vectorData_test/temp/")
    output_dir.mkdir(exist_ok=True, parents=True)
    mask = test(model, loader, output_dir, idx, loader.dataset[0].y)




    # model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)
    # print("model = ", model)
    # print("optimizer = ", optimizer)
    # print("start_epoch = ", start_epoch)
    # print("valid_loss_min = ", valid_loss_min)
    # print("valid_loss_min = {:.6f}".format(valid_loss_min))
    


    

