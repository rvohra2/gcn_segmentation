import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
import collections
import os
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import numpy_groupies as npg
import cairosvg
from PIL import Image
import io
import scipy.sparse as sp
from PIL import ImageOps
import xml.etree.ElementTree as et
from skimage import color, segmentation
from skimage.segmentation import mark_boundaries
import torch_geometric.nn as geometric_nn
import torch.nn.functional as F
import networkx as nx
from convert_svg import render_svg
from gcn_model import GCNs
import torchvision.models as models

###Working better for 500 epoch
Epochs = 500

model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def normalize(adj):
    rowsum = np.array(adj.sum(1))
    adjpow = np.power(rowsum, -1).flatten()
    adjpow[np.isinf(adjpow)] = 0.
    r_mat_inv = sp.diags(adjpow)
    adj = r_mat_inv.dot(adj)
    return adj


def get_xy(x_list,y_list):
    for x in x_list:
        for y in y_list:
            yield (x,y)

def adjacency_matrix(labels, conn=4, loop=True, max_label=-1):
    """ labels: (B, H, W) or (B,1,H,W) torch.uint8 """
    assert conn in [4,8], f"conn must be either 4 or 8 connectivity"

    labels = torch.from_numpy(labels)
    if len(labels.shape) == 4:
        assert labels.size(1) == 1, f"labels must be of size (B,H,C) or (B,1,H,C): {labels.shape}"
        labels = labels.squeeze(1)

    H, W = labels.shape
    N = H*W
    adj = torch.zeros((N, N), dtype=torch.float32).to(labels.device)
    max_label = torch.max(labels) + 1 if max_label == -1 else torch.tensor(max_label)
    
    for (y,x) in get_xy(list(range(0, H, 1)), list(range(0, W, 1))):

        #top-left
        node=y*W + x
        i = labels[y,x]
        #print(f"node={node+1} ({x},{y}) -> {i}")
      
        if i.sum() == 0: continue 

        #itself
        if loop:
            adj[node, node] = torch.min((i > 0) * i, max_label)
        #print(f"node={node+1} (-> itself) ({x},{y}) -> {i > 0} * {i} -> {(i > 0) * i}")
    
        #right
        if x+1 < W:
            nn = i == labels[y,x+1]
            #print(f"node={node+1} (->right) ({x+1},{y}) -> {i} -> {nn*i}")
            if nn.sum():
                label = torch.min(nn*i, max_label)
                adj[node, node+1] = label 
                adj[node+1, node] = label

        #down
        if y+1 < H:
            nn = i == labels[y+1,x]
            #print(f"node={node+1} (->down) ({x},{y+1}) -> {i} -> {nn*i}")
            if nn.sum():
                j = (y+1)*W + x
                label = torch.min(nn*i, max_label)
                adj[node, j ] = label 
                adj[j, node]  = label
       
        #TODO: check this
        if conn == 8:
            #down diagonal
            if y+1 < H and x+1 < W:
                nn = i == labels[y+1,x+1]
                if nn.sum():
                    j = (y+1)*W + x
                    label = torch.min(nn*i, max_label)
                    adj[node+1, j] = label
                    adj[j, node+1] = label 
           
            #up diagonal
            if y-1 >= 0 and x+1 < W:
                #i = labels[:, y-1, x]
                nn = i == labels[y-1,x+1]
                if nn.sum():
                    j = (y-1)*W + x
                    label = torch.min(nn*i, max_label)
                    adj[j , node+1] = label 
                    adj[node+1, j]  = label 

        #print(adj)


    return adj

def create_edge_list(adj):
    edge_list = [[], []]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                edge_list[0].append(i)
                edge_list[1].append(j)
    edge_list = np.array(edge_list)
    return edge_list

def create_features(image, segmentation):
    # total = 0
    # features = []
    # max_dim = -1
    # for i in range(np.amax(segmentation)+1):
    #     indice = np.where(segmentation==i)
    #     features.append(image[indice[0], indice[1]].flatten().tolist())
    #     if len(features[-1]) > max_dim:
    #         max_dim = len(features[-1])

    # for i in range(len(features)):
    #     features[i].extend([0] * (max_dim-len(features[i])))


    image = Image.fromarray(image)
    t_img = transforms(image)
    my_embedding = torch.zeros(512)
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())
    
    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():
        model(t_img.unsqueeze(0))
    h.remove()
    features = np.array(my_embedding)
    features = np.array(features)
    features_norm = (features - np.min(features)) / (np.max(features) - np.min(features))
    features_norm = np.reshape(features_norm, (2,-1))
    return features_norm

def create_target(segmentation, target_mask):
    y = []
    for i in range(np.amax(segmentation)+1):
        indices = np.where(segmentation==i)
        patch = target_mask[indices[0], indices[1]]
        max_label = collections.Counter(patch).most_common()[0][0]

        y.append(max_label)
    return np.array(y)

def select_mask_color_test(cls, colors):
    background_color = [0, 0, 0]
    if cls == 0:
        return background_color
    else:
        return colors[cls]

def svg_to_png(svg):
    
    image = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
    image = ImageOps.invert(image)
    image.thumbnail((64,64), Image.ANTIALIAS)
    assert (image.size == (64, 64))
    return image

def _preprocess(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return color.gray2rgb(
            np.reshape(image, (image.shape[0], image.shape[1])))
    else:
        return image

def slic(image, num_segments, compactness=10, max_iterations=20, sigma=0):
    image = _preprocess(image)
    return segmentation.slic(image, num_segments, compactness, max_iterations,
                             sigma, start_label=0)

def slic_fixed(num_segments, compactness=1, max_iterations=2, sigma=0):
    def slic_image(image):
        return slic(image, num_segments, compactness, max_iterations, sigma)

    return slic_image

def test_loader(max_dim):
    features_list = []
    edge_list = []
    targets = []
    ###Change filename number
    svg_name = os.path.join('/home/rhythm/notebook/vectorData_test/svg/1.svg')
    with open(svg_name, 'r') as f_svg:
        svg = f_svg.read()

    ###Uncomment for cat/baseball dataset
    # num_paths = svg.count('polyline')

    # for i in range(1, num_paths + 1):
    #     svg_xml = et.fromstring(svg)
    #     #svg_xml[1] = svg_xml[i]
    #     #del svg_xml[2:]
    #     svg_one = et.tostring(svg_xml, method='xml')

    #     # leave only one path
    #     y_png = cairosvg.svg2png(bytestring=svg_one)
    #     y_img = Image.open(io.BytesIO(y_png))
    #     mask = (np.array(y_img)[:, :, 3] > 0)
    #     mask = mask.astype(np.uint8)

    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.show()


    ###Uncomment for line drawing dataset
    num_paths = len(et.fromstring(svg)[0])
    for i in range(num_paths):
        svg_xml = et.fromstring(svg)
        # svg_xml[0][0] = svg_xml[0][i]
        # del svg_xml[0][1:]
        svg_one = et.tostring(svg_xml, method='xml')

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one)
        y_img = Image.open(io.BytesIO(y_png))
        mask = (np.array(y_img)[:, :, 3] > 0)
        mask = mask.astype(np.uint8)

    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    image = svg_to_png(svg)
    image = np.asarray(image)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    ###Need to fine tune
    segmentation_algorithm = slic_fixed(100, compactness=1, max_iterations=10, sigma=0)
    segmentation = segmentation_algorithm(image)

    seg_img = mark_boundaries(image, segmentation)
    # fig = plt.figure("Superpixels -- %d segments" % (100))
    # plt.imshow(seg_img)
    # plt.show()

    target_mask = mask
    #target_mask = np.asarray(target_mask.resize((200,200)))[:, :, 1]
    #y = Variable(torch.from_numpy(create_target(segmentation, target_mask)))
    target_mask = np.array(target_mask)
    y = Variable(torch.from_numpy(target_mask))
    num_instance = np.max(target_mask)+1
    targets.append(y)

    adj = adjacency_matrix(target_mask)
    adj = np.asarray(adj)
    #adj = np.array(adj.todense())
    adj = torch.from_numpy(adj).type(torch.FloatTensor)
    
    edge_x =[]
    #edge_x = Variable(torch.from_numpy(create_edge_list(adj))).cuda()
    features = Variable(torch.from_numpy(create_features(image, segmentation))).type(torch.cuda.FloatTensor)
    
    datasets = []
    datasets.append(Data(features, edge_x, y=targets))

    return DataLoader(datasets, batch_size=1), adj, image, segmentation, target_mask

loader, adj, image, segmentation, target_mask = test_loader(max_dim=-1)

def test(model, adj, num_instance_label, max_dim):
    
    loader, adj,image, segmentation, target_mask = test_loader(max_dim)
    # color
    num_colors = num_instance_label
    colors = np.array([colorsys.hsv_to_rgb(h, 0.8, 0.8)
                   for h in np.linspace(0, 1, num_colors)]) * 255
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    node_num = len(np.unique(segmentation))
    correct = 0
    with torch.no_grad(): 
        for data in loader:
            y = data.y
            y = y[0].type(torch.cuda.LongTensor)
            x = data.x
            x = x.cpu()

            logits = model(data)
            logp = F.log_softmax(logits, 1)
            pred = logp.max(1, keepdim=True)[1].cuda()
            for v in range(0, node_num):
                cls = pred[v][0].cpu().detach().numpy()
                mask_color = select_mask_color_test(cls, colors)
                
                mask[segmentation == v] = mask_color
            correct += pred.eq(y.view_as(pred)).sum().item()
        data_num = len(y)  
        print('\n Accuracy : {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))
        return image, mask, target_mask

def train(model, optimizer, loader, adj):
    all_logits = []
    for epoch in range(Epochs):
        model.train()
        loss = 0
        for data in loader:
            #print("data = {}".format(data))
            y = data.y
            y = y[0].type(torch.cuda.LongTensor)
            x = data.x
            x = x.cpu()
            
            optimizer.zero_grad()
            output = model(x, adj).cuda()
            all_logits.append(output.detach())
            #output = output.transpose(0, 1)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            # logits = model(x, adj)
            # all_logits.append(logits.detach())
            # logp = F.log_softmax(logits, 1).cuda()
            # loss = F.nll_loss(logp, y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    return all_logits

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN(nfeat=loader.dataset[0].x.shape[1],nhid=1024,nclass=2,dropout=0.5).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


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

###Working better for nhid=1024, dropout=0.5, lr=0.01
model = GCNs(nfeat=loader.dataset[0].x.shape[1],nhid=1024,nclass=2,dropout=0.5)
optimizer = optim.Adam(model.parameters(), 0.01)
all_logits = train(model, optimizer, loader, adj)

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



adj = adjacency_matrix(segmentation)
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

del segmentation
from skimage import color, segmentation
image, mask, target_mask = test(model, adj, 2, max_dim=-1)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(mask)
plt.subplot(1,3,3)
plt.imshow(target_mask)
plt.show()

mask = mask.mean(axis=2)
mask = (mask != 0)
###Change filename number
im = render_svg(image, mask, node_num,  "/home/rhythm/notebook/vectorData_test/temp/1_test.svg")


    




