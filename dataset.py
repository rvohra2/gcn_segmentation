import torch
import os
import cairosvg
import xml.etree.ElementTree as et
import io
from PIL import Image
import numpy as np
from PIL import ImageOps
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from pathlib import Path
from slic_segm import slic_fixed
from utils import segmentation_adjacency, create_edge_list, create_features, create_target

def svg_to_png(svg):
    image = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
    image = ImageOps.invert(image)
    assert (image.size == (64,64))
    return image

data_dir = Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/ch/")
def BaseDataset():
    with (data_dir / "{:s}.txt".format('train')).open("r") as f:
            ids = [_.strip() for _ in f.readlines()]
    return ids

def ChineseDataset(ids, idx):
    masks = []
    svg_name = data_dir / "svg" / "{:s}.svg".format(ids[idx])
        
    with svg_name.open('r') as f_svg:
        svg = f_svg.read()

    num_paths = len(et.fromstring(svg)[0])
        
    for i in range(num_paths):
        svg_xml = et.fromstring(svg)
        svg_xml[0][0] = svg_xml[0][i]
        del svg_xml[0][1:]
        svg_one = et.tostring(svg_xml, method='xml')

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one)
        y_img = Image.open(io.BytesIO(y_png))
        mask = (np.array(y_img)[:, :, 3] > 0)
        masks.append(mask.astype(np.uint8))
    
    return svg_to_png(svg), masks, num_paths

def QuickDrawDataset(ids, idx):
    masks = []
    svg_name = data_dir / 'train' / "{:s}.svg".format(ids[idx])
    with svg_name.open('r') as f_svg:
        svg = f_svg.read()

    num_paths = svg.count('polyline')

    for i in range(1, num_paths + 1):
        svg_xml = et.fromstring(svg)
        # svg_xml[1] = svg_xml[i]
        # del svg_xml[2:]
        svg_one = et.tostring(svg_xml, method='xml')

        # leave only one path
        y_png = cairosvg.svg2png(bytestring=svg_one)
        y_img = Image.open(io.BytesIO(y_png))
        mask = (np.array(y_img)[:, :, 3] > 0)
        #masks.append(mask.astype(np.uint8))

    return svg_to_png(svg), mask.astype(np.uint8), num_paths

# svg_name = os.path.join('/home/rhythm/notebook/vectorData_test/svg/12.svg')
# with open(svg_name, 'r') as f_svg:
#     svg = f_svg.read()
