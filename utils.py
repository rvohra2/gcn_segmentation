import numpy as np
import scipy.sparse as sp
import numpy_groupies as npg
import collections
import cairosvg
import io
from PIL import Image, ImageOps
import torch
import shutil
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import functional as Fa
import random
from torchvision.transforms import transforms as T

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
    #adj_norm = (adj - np.min(adj)) / (np.max(adj) - np.min(adj))
    #print(adj_norm.min(), adj_norm.max())
    return adj

def create_edge_list(adj):
    x = []
    edge = np.where(adj!=0)
    x.append(edge)
    x = np.array(x)[0]

    # edge_list = [[], []]
    # for i in range(adj.shape[0]):
    #     for j in range(adj.shape[1]):
    #         if adj[i][j] != 0:
    #             edge_list[0].append(i)
    #             edge_list[1].append(j)
    # edge_list = np.array(edge_list)
    # print((x == edge_list).all())
    #print(edge_list.min(), edge_list.max())
    return x

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
        
        features[i].extend([0] * (500-len(features[i])))
        

    features = np.array(features)
    #features = features.reshape(-2, 27)
    #features_norm = (features - np.min(features)) / (np.max(features) - np.min(features))
    #print(features_norm.min(), features_norm.max())
    
    return features

def create_target(segmentation, target_mask, num_paths):
    y = []
    #print(np.amax(segmentation))
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

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min

def focal_loss(x, y):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,D].
        y: (tensor) sized [N,].
    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2

    t = F.one_hot(y, 30)  # [N,21]
    #t = t[:,1:]  # exclude background
    t = Variable(t).type(torch.cuda.FloatTensor)
    
    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction='mean')




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = Fa.resize(image, size)
        target = np.resize(target, size)
        return image, target

class RandomResizeCrop(object):
    def __init__(self, prob=0.5, crop_size=(64, 64), max_scale=1.25):
        assert max_scale >= 1.0
        self.prob = prob
        self.max_scale = max_scale
        self.crop_size = crop_size
        self.cropper = T.RandomCrop(crop_size)

    def __call__(self, image, target):
        if random.random() < self.prob:
            # resize
            h = int(self.crop_size[0] * random.uniform(1.0, self.max_scale))
            w = int(self.crop_size[1] * random.uniform(1.0, self.max_scale))
            image = F.resize(image, (h, w))
            target = target.resize(image.size)

            # random crop
            i, j, h, w = self.cropper.get_params(image, self.crop_size)
            image = F.crop(image, i, j, h, w)
            target = target.crop((j, i, j + w, i + h))
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            
            image = Fa.hflip(image)
            target = np.flip(target, 1)
            
            #target = target.transpose(0)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        
        return Fa.to_tensor(image), target

def build_transforms(is_train=True):
    if is_train:
        min_size = 128
        max_size = 128
        crop_prob = 0.0
        flip_prob = 0.5
        transform_list = [
            Resize(min_size, max_size),
            RandomResizeCrop(crop_prob, (min_size, min_size)),
            RandomHorizontalFlip(flip_prob),
        ]
    else:
        transform_list = []

    transform_list.append(ToTensor())
    transform = Compose(transform_list)
    return transform