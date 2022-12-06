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
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import random
import math

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
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
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
        
        features[i].extend([0] * (1200-len(features[i])))

    
    features1 = np.array(features)    
    print(features1.shape)
    features = np.array(features) /255.
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
    y = np.array(y)
    return y

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
    instance1_color = [0, 255, 0]
    instance2_color = [0, 0, 255]
    instance3_color = [255, 0, 0]
    instance4_color = [255, 0, 255]
    instance5_color = [29, 32, 136]
 
    
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
    else:
        return instance5_color 

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

    t = F.one_hot(y, 40)  # [N,21]
    #t = t[:,1:]  # exclude background
    t = Variable(t).type(torch.cuda.FloatTensor)
    
    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach(), reduction='mean')

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([.25, .75])
        self.gamma = 2

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def accuracy(output, labels):
    preds = output.type_as(labels)
    correct = preds.eq(labels)
    correct = correct.sum()
    return correct / (labels.size(0)*labels.size(1))

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


class Stroke(object):
    class_counter= 0

    def __init__(self, points, image):

        self.points = points
        self.image = image
        self.starting_point, self.ending_point = self.find_ending_points()
        self.index = Stroke.class_counter
        Stroke.class_counter += 1

    def draw_strokes3(self, image, points):
        if len(image.shape) >2:
            h, w, d = image.shape
        else :
            h, w= image.shape
        blank_image = np.zeros((h, w), np.uint8)

        for p in points:
            blank_image[p[1], p[0]] = 255

        return blank_image

    def distance(self, pixel_1, pixel_2):
        delta_x = (pixel_1[0] - pixel_2[0]) ** 2
        delta_y = (pixel_1[1] - pixel_2[1]) ** 2
        return (delta_x + delta_y) ** 0.5

    def far_away_end_points(self, ending_points):
        max_dist = -10
        ending_points_ = None
        for p in ending_points:
            for p2 in ending_points:
                d = self.distance(p, p2)
                if d > max_dist:
                    ending_points_ = [p, p2]
                    max_dist = d
        return ending_points_



    def compareN(self, neighbourhood):
        possible = [np.array([[0, 255, 255], [0, 255, 0], [0, 0, 0]]),
                    np.array([[255, 255, 0], [0, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [255, 255, 0]]),
                    np.array([[0, 0, 0], [0, 255, 0], [0, 255, 255]]),
                    np.array([[0, 0, 255], [0, 255, 255], [0, 0, 0]]),
                    np.array([[255, 0, 0], [255, 255, 0], [0, 0, 0]]),
                    np.array([[0, 0, 0], [255, 255, 0], [255, 0, 0]]),
                    np.array([[0, 0, 0], [0, 255, 255], [0, 0, 255]])
                    ]
        for p in possible:

            c = neighbourhood == p
            if c.all():
                return True
        return False

    def find_ending_points(self):
        d = self.draw_strokes3(self.image, self.points)
        ending_points = []
        for p in self.points:
            neighbourhood = list()
            neighbourhood[:] = d[p[1] - 1: p[1] + 2, p[0] - 1: p[0] + 2]
            neighbours = np.argwhere(neighbourhood)
            #print('neighbours', neighbours, 'len neighbours', len(neighbours))
            if len(neighbours) <= 2:
                ending_points.append(p)
            elif len(neighbours) == 3:
                if  self.compareN(neighbourhood):
                    ending_points.append(p)
        # returns the two ending points that are more far away
        #print("points", self.points)
        #print("ending_points", ending_points)
        # if no ending points were found we have a close stroke (i.e. a perfectly closed circle)
        if not ending_points:
            return (self.points[0], self.points[1])

        real_ending_points = self.far_away_end_points(ending_points)
        return (real_ending_points[0], real_ending_points[-1])


def threshold_image(image, min_p = 10, max_p = 255):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(grayscaled, min_p, max_p, cv2.THRESH_BINARY)
    return threshold


def getSkeleton (image):
    # Applies a skeletonization algorithm, thus, transforming all the strokes inside the sketch into one pixel width lines
    threshold = threshold_image(image)
    threshold = cv2.bitwise_not(threshold)
    threshold[threshold == 255] = 1
    skeleton = skeletonize(threshold)
    skeleton = img_as_ubyte(skeleton)
    return skeleton

def showImage(image, method = 'plt'):
    if method == 'plt':
        try:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
        except cv2.error:
            plt.imshow(image)
            plt.show()

    else:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def scan_8_pixel_neighbourhood(skeleton_image, pixel):
    """
    :param skeleton_image: skeleton image
    :param pixel: a tuple of the type (x, y)
    :return: a matrix indicating the indexes of the neighbouring pixels of the input pixel
    """
    if inside_image(pixel, skeleton_image):
        skeleton_image = skeleton_image.copy()
        neighbourhood = skeleton_image[pixel[1] - 1: pixel[1] + 2, pixel[0] - 1: pixel[0] + 2]
        neighbourhood[1,1] = 0
        neighbours = np.argwhere(neighbourhood)
        return neighbours
    else:
        return []

def find_top_left_most_pixel(skeleton_image, processing_index = 0):
    """
    Expects an skeletonized image (binary image with one-pixel width lines)
    """
    for y in range(processing_index, skeleton_image.shape[0], 1):
        for x in range(0, skeleton_image.shape[1]):

            if skeleton_image[y, x] == 255 :
                return (x,y)
    return None

def inside_image(pixel, image):
    """Checks whether a pixel is inside the image space"""
    h, w = image.shape
    if (pixel[1] - 1 >= 0) and (pixel[1] + 1 <= h - 1) and (pixel[0] - 1 >= 0) and (pixel[0] + 1 <= w - 1):
        return True
    else:
        return False


def extend_line(P1, P2 , offset = 100000):
    x1, y1 = P1
    x2, y2 = P2

    delta_x = x2 - x1
    delta_y = y2 - y1

    new_x1 = x1 - delta_x * offset
    new_y1 = y1 - delta_y * offset

    new_x2 = x2 + delta_x * offset
    new_y2 = y2 + delta_y * offset

    return ((new_x1, new_y1), (new_x2, new_y2))


def determine_side(P1, P2, P3):

    """Determines whether the point P3 is to left or to the right side of the line formed
       by the points P1 and P2
    """
    P1, P2 = extend_line(P1, P2)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3

    d3 = (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)

    # d1 is calculated for a point that we know lies on the left side of the line
    d1 = ((x1 - 1) - x1)*(y2 - y1) - (y1 - y1)*(x2 - x1)
    sign = lambda a: 1 if a > 0 else -1 if a < 0 else 0

    if sign(d3) == sign(d1):
        return "left"
    else:
        return "right"

def inner_angle(P1, P2, P3):
    """Computes the inner product formed by the lines generated from
       (P1(x1, y1), P2(x2, y2) and P2(x2, y2), P3(x3, y3))
       P2 is shared by both lines, hence it represents the point of ambiguity
    """
    side = determine_side(P1, P2, P3)
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = P3
    dx21 = x1 - x2
    dx31 = x3 - x2
    dy21 = y1 - y2
    dy31 = y3 - y2
    m12 = (dx21 * dx21 + dy21 * dy21) ** 0.5
    m13 = (dx31 * dx31 + dy31 * dy31) ** 0.5
    theta_radians = math.acos((dx21 * dx31 + dy21 * dy31) / (m12 * m13))
    theta_degrees = theta_radians * 180 / math.pi

    if side == "left":
        theta_degrees = 360 - theta_degrees

    return theta_degrees


def local_solver(P1, P2, neighbours):
    """
    from a set of neighbouring pixels selects the one with the minimum angular deviation
    from the direction given by the last two pixels of the stroke history (P1,P2).
    """
    minimum_angle = 100000
    selected_pixel = None
    delta = {(0,0): (-1,-1), (0,1): (-1,0), (0,2): (-1,1),
            (1,0): (0,-1), (1,1): (0,0), (1,2): (0,1),
            (2, 0): (1, -1), (2,1): (1,0), (2,2): (1,1)}
    for n in neighbours:
        delta_y, delta_x = delta[tuple(n)]
        P3 = (P2[0] + delta_x, P2[1] + delta_y)
        angle = inner_angle(P1, P2, P3)
        if angle < minimum_angle:
            selected_pixel = P3
            minimum_angle = angle
    return selected_pixel


def draw_strokes(image, strokes, colors=[(0,255,0)]):
    if len(image.shape)>2:
        h, w, d = image.shape
    else:
        h,w = image.shape
        d=3
    blank_image = np.zeros((h, w, d), np.uint8)
    color_index = 0
    for stroke in strokes:
        for p in stroke.points:
            cv2.circle(blank_image, p, 1, colors[color_index%len(colors)], -1)
            #cv2.imwrite(f'{color_index}.bmp', blank_image)
        color_index += 1

    return blank_image


def distance(pixel_1, pixel_2):
    delta_x = (pixel_1[0] - pixel_2[0])**2
    delta_y = (pixel_1[1] - pixel_2[1]) ** 2
    return (delta_x + delta_y)**0.5


def stroke_distance(stroke_1, stroke_2):
    """Returns the minimum distance between two strokes"""
    s_1 = stroke_1.starting_point
    e_1 = stroke_1.ending_point
    s_2 = stroke_2.starting_point
    e_2 = stroke_2.ending_point
    set_1 = [s_1, e_1]
    set_2 = [s_2, e_2]
    frontier = None
    minimum_distance = 1000000
    for p in set_1:
        for p2 in set_2:
            p_dist = distance(p, p2)
            if p_dist < minimum_distance:
                minimum_distance = p_dist
                frontier = (p,p2)
    return minimum_distance, frontier


def generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length = 10):
    """Generates an stroke until ambiguity, unless the current length is less than a predefined length threshold"""
    former_skeleton = skeleton_image.copy()
    ambiguity_pixel = None
    stroke_history = []
    stroke_history.append(pixel)
    all_possibilities = []
    skeleton_image[pixel[1], pixel[0]] = 0
    ambiguity_solved = True
    delta = {(0,0): (-1,-1), (0,1): (-1,0), (0,2): (-1,1),
            (1,0): (0,-1), (1,1): (0,0), (1,2): (0,1),
            (2, 0): (1, -1), (2,1): (1,0), (2,2): (1,1)}

    while (len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) > 0) or not ambiguity_solved:
        #print(ambiguity_solved)
        if ambiguity_pixel:
            neighbours_ap = scan_8_pixel_neighbourhood(skeleton_image, ambiguity_pixel)
            #print("ambiguity pixel", ambiguity_pixel, "has", len(neighbours_ap), "neighbours")
            #print(neighbours_ap)
            if len(neighbours_ap) == 0:
                #print("hiiii", ambiguity_pixel)
                ambiguity_solved = True

        if len(scan_8_pixel_neighbourhood(skeleton_image, pixel)) == 0 and ambiguity_solved==False:
            # added

            skeleton_image[pixel[1], pixel[0]] = 0
            pixel = ambiguity_pixel
            all_possibilities.append(stroke_history)
            stroke_history = []

        # comparing with the new pixel
        neighbours = scan_8_pixel_neighbourhood(skeleton_image, pixel)


            # added check
        #if len(neighbours) == 0:


        if len(neighbours) == 1:
            delta_y, delta_x = delta[tuple(neighbours[0])]
            pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
            stroke_history.append(pixel)
            #print("Stroke History", stroke_history)
            skeleton_image[pixel[1], pixel[0]] = 0
        elif len(stroke_history) < minimum_length and len(neighbours) > 0:
            if len(stroke_history) < 2:
                #print("neighbours XD XD", neighbours)
                #print("neighbours XD", tuple(neighbours[0]))
                delta_y, delta_x = delta[tuple(neighbours[0])]
                pixel = (pixel[0] + delta_x, pixel[1] + delta_y)
                stroke_history.append(pixel)
                #print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
            else:
                P1 = stroke_history[-2]
                P2 = stroke_history[-1]

                pixel = local_solver(P1, P2, tuple(neighbours))
                stroke_history.append(pixel)
                #print("Stroke History", stroke_history)
                skeleton_image[pixel[1], pixel[0]] = 0
        else:
            #it is large enough so it can be compared later, hence we add it

            all_possibilities.append(stroke_history)
            #print("all_possibilities", stroke_history)
            stroke_history = []
            # we must go back to the original ambiguity
            if ambiguity_pixel:
                pixel = ambiguity_pixel
            else:
                ambiguity_pixel = pixel
                ambiguity_solved = False
                #print("ambiguity pixel", ambiguity_pixel, ambiguity_solved)

    if len(stroke_history)>=12:
        all_possibilities.append(stroke_history)

    all_strokes = [Stroke(points, former_skeleton) for points in all_possibilities if len(points) > 4]

    return all_strokes, former_skeleton


def generate_strokes2(skeleton_image):
    all_strokes = []
    while True:
        pixel = find_top_left_most_pixel(skeleton_image, processing_index=0)
        if  pixel == None:
            break

        strokes, former_skeleton = generate_strokes_until_ambiguity(skeleton_image, pixel, minimum_length=10)
        # if len(strokes) == 0:
        #     break

        for s in strokes:
            all_strokes.append(s)

    return  all_strokes, former_skeleton



def points_principal_component(points):
    """The points list must have  a length of 12"""
    x = np.array([p[0] for p in points])
    x_mean = x.mean()
    y = np.array([p[1] for p in points])
    y_mean  = y.mean()
    principal_component = np.sum((x - x_mean) * (y - y_mean))/(11)
    return principal_component


def best_stroke(former_stroke, possible_strokes ):
    """Returns best stroke to be merged inside possible_strokes according to the principal component"""
    best_stroke = None
    index = None
    minimum_difference = 10000000000000000
    pc_fs = points_principal_component(former_stroke.points[-12:])
    for index, ps in enumerate(possible_strokes):
        pc = points_principal_component(ps.points[0:12])
        diff = (pc_fs - pc)**2
        if diff < minimum_difference:
            minimum_difference = diff
            best_stroke = ps
    return best_stroke, index


def fill_stroke_gap(frontier):
    p1, p2 = frontier
    new_points_x = list(range(p1[0], p2[0] + 1, 1))
    new_points_y = list(range(p1[1], p2[1] + 1, 1))
    new_points = list(zip(new_points_x, new_points_y))
    return new_points


def alternative_single_merge(former_stroke, possible_strokes, image):
    """merges the former stroke with the best stroke within the possibilities"""
    possibilities = []
    strokes_to_be_erased = []


    for ps in possible_strokes:
        d,frontier = stroke_distance(former_stroke, ps)
        if d < 10:
            possibilities.append(ps)
    best_stroke_, index = best_stroke(former_stroke, possibilities)
    if best_stroke_:
        d, frontier = stroke_distance(former_stroke, best_stroke_)
        points_to_add = fill_stroke_gap(frontier)
        strokes_to_be_erased.append(best_stroke_.index)
        strokes_to_be_erased.append(former_stroke.index)
        #print(points_to_add)
        new_stroke = Stroke(best_stroke_.points + points_to_add + former_stroke.points, image)

        return new_stroke,strokes_to_be_erased
    else:
        return former_stroke,strokes_to_be_erased


def multiple_merge(all_strokes, image):

    former_stroke = all_strokes[0]
    to_compare = all_strokes[1:]
    while True:
        former_stroke, strokes_to_be_erased = alternative_single_merge(former_stroke, to_compare, image)
        to_compare = [tc for tc in to_compare if tc.index not in strokes_to_be_erased]
        if len(strokes_to_be_erased) == 0:
            break
    return former_stroke, to_compare


def generate_final_strokes(image):
    skeleton_image = getSkeleton(image)
    all_strokes, _ = generate_strokes2(skeleton_image)
    final_strokes = []
    while True:
        #print("Generating Strokex")
        former_stroke, comparision_strokes = multiple_merge(all_strokes,image)
        if former_stroke:
            final_strokes.append(former_stroke)
        if len(comparision_strokes) == 0:
            break

        all_strokes = comparision_strokes
    return final_strokes