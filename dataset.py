import torch
import os
import cairosvg
import xml.etree.ElementTree as et
import io
from PIL import Image
import numpy as np
from PIL import ImageOps
from pathlib import Path
from utils import build_transforms
import sys
import matplotlib.pyplot as plt

def get_dataset(dataset="ch", subset="test", is_train=True):
 
    datasets_dict = {
        "ch": ChineseDataset,
        #"kanji": KanjiDataset,
        #"line": LineDataset,
        "cat": QuickDrawDataset,
        #"baseball": QDrawDataset,
        #"multi": QDrawDataset,
    }
    root = (Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/qdraw") / dataset)
    transforms = build_transforms(is_train)
    return datasets_dict[dataset](root, subset, transforms)

root = "../data"
dataset = "ch"
#data_dir = Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/ch/")

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transforms=None, size=64):
        self.data_dir = data_dir
        self.split = split

        with (self.data_dir / "{:s}.txt".format(split)).open("r") as f:
                self.ids = [_.strip() for _ in f.readlines()]
        self.transforms = transforms
    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        image, out, num_paths = self.get_groundtruth(idx)
        
        if self.transforms:
            image, out = self.transforms(image, out)
            image = image.permute(1, 2, 0)
            return image, out, num_paths
            
        return image, out, num_paths

    def svg_to_png(self, svg):
        image = cairosvg.svg2png(bytestring=svg)
        image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
        image = ImageOps.invert(image)
        #assert (image.size == (128,128))
        return image

class ChineseDataset(BaseDataset):

    def __init__(self, data_dir, split, transforms=None, size=64):
        super().__init__(data_dir, split, transforms, size)

    def get_groundtruth(self, idx):
        masks = []
        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
            
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
        
        return self.svg_to_png(svg), (masks), num_paths

class QuickDrawDataset(BaseDataset):

    def __init__(self, data_dir, split, transforms=None, size=128):
        super().__init__(data_dir, split, transforms, size)

    def get_groundtruth(self, idx):
        masks = []
        svg_name = self.data_dir / self.split / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        num_paths = svg.count('polyline')
        out = np.zeros(shape=(128,128))

        for i in range(1, num_paths + 1):
            svg_xml = et.fromstring(svg)
            svg_xml[1] = svg_xml[i]
            del svg_xml[2:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            y_img.thumbnail((128,128))
            # plt.imshow(y_img)
            # plt.show()
            
            #mask = (np.array(y_img)[:, :, 3] > 0)
            mask = (np.array(y_img)[:, :, 3] > 0).astype(np.uint8)
            #print(np.unique(mask))
            mask = np.where(mask == 1, i, 0)
            
            
            
            #print(mask)
            out += mask
            out = np.asarray(out, dtype=int)
            #print('i: ',i, 'mask: ', out.max())
            out[out>i] = i


        return self.svg_to_png(svg), out, num_paths

# svg_name = os.path.join('/home/rhythm/notebook/vectorData_test/svg/12.svg')
# with open(svg_name, 'r') as f_svg:
#     svg = f_svg.read()
