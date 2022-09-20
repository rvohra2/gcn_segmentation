import numpy as np
import scipy.sparse as sp
import numpy_groupies as npg
import collections
import cairosvg
import io
from PIL import Image, ImageOps

def normalize(adj):
    rowsum = np.array(adj.sum(1))
    adjpow = np.power(rowsum, -1).flatten()
    adjpow[np.isinf(adjpow)] = 0.
    r_mat_inv = sp.diags(adjpow)
    adj = r_mat_inv.dot(adj)
    return adj

def segmentation_adjacency(segmentation, connectivity=8):
    """Generate an adjacency matrix out of a given segmentation."""

    ###Working better for connectivity=8
    assert connectivity == 4 or connectivity == 8

    # Get centroids.
    idx = np.indices(segmentation.shape)
    ys = npg.aggregate(segmentation.flatten(), idx[0].flatten(), func='mean')
    xs = npg.aggregate(segmentation.flatten(), idx[1].flatten(), func='mean')
    ys = np.reshape(ys, (-1, 1))
    xs = np.reshape(xs, (-1, 1))
    points = np.concatenate((ys, xs), axis=1)

    # Get mass.
    nums, mass = np.unique(segmentation, return_counts=True)
    n = nums.shape[0]

    # Get adjacency (https://goo.gl/y1xFMq).
    tmp = np.zeros((n, n), bool)

    # Get vertically adjacency.
    a, b = segmentation[:-1, :], segmentation[1:, :]
    
    tmp[a[a != b], b[a != b]] = True

    # Get horizontally adjacency.
    a, b = segmentation[:, :-1], segmentation[:, 1:]
    tmp[a[a != b], b[a != b]] = True

    # Get diagonal adjacency.
    if connectivity == 8:
        a, b = segmentation[:-1, :-1], segmentation[1:, 1:]
        tmp[a[a != b], b[a != b]] = True

        a, b = segmentation[:-1, 1:], segmentation[1:, :-1]
        tmp[a[a != b], b[a != b]] = True

    result = tmp | tmp.T
    result = result.astype(np.uint8)
    adj = sp.coo_matrix(result)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
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
    total = 0
    features = []
    max_dim = -1
    
    for i in range(np.amax(segmentation)+1):
        indice = np.where(segmentation==i)
        features.append(image[indice[0], indice[1]].flatten().tolist())
        if len(features[-1]) > max_dim:
            max_dim = len(features[-1])
    
    for i in range(len(features)):
        
        features[i].extend([0] * (300-len(features[i])))
        

    features = np.array(features)
    #features = features.reshape(-2, 27)
    features_norm = (features - np.min(features)) / (np.max(features) - np.min(features))
    
    return features_norm

def create_target(segmentation, target_mask, num_paths):
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
    image.thumbnail((128,128), Image.ANTIALIAS)
    #assert (image.size == (32,32))
    return image

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
    instance1_color = [143, 195, 131]
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

def create_mask(img, segmentation, node_num, epoch, pos, all_logits):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    for v in range(0, node_num):
        pos[v] = all_logits[epoch][v].cpu().numpy()
        
        cls = pos[v].argmax()
        
        mask_color = select_mask_color(cls)
        #print(mask_color)
        mask[segmentation == v] = mask_color
        
    return img ,mask