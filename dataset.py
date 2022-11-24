import torch
import os
import cairosvg
import xml.etree.ElementTree as et
import io
from PIL import Image
import numpy as np
from PIL import ImageOps
from pathlib import Path
from utils import build_transforms, segmentation_adjacency, create_edge_list, create_features, create_target
import sys
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data
from slic_segm import slic_fixed
from torch.autograd import Variable
import os.path as osp
import config
import cv2

def get_dataset(dataset="ch", subset="test", is_train=True):
 
    datasets_dict = {
        "ch": ChineseDataset,
        #"kanji": KanjiDataset,
        #"line": LineDataset,
        "cat": QuickDrawDataset,
        #"baseball": QDrawDataset,
        #"multi": QDrawDataset,
    }
    root = (Path(config.DATASET_PATH) / dataset)
    transforms = build_transforms(is_train)
    return datasets_dict[dataset](root, subset, transforms)


class BaseDataset(Dataset):
    def __init__(self, root, split, transforms=None, size=64):
        self.root_dir = root
        self.data_dir = Path(config.ROOT_PATH)
        self.split = split

        with (self.root_dir / "{:s}.txt".format(split)).open("r") as f:
                self.ids = [_.strip() for _ in f.readlines()]
        self.transforms = transforms
    
        if not os.path.isdir(self.data_dir / self.split):
            os.makedirs(self.data_dir / self.split)

        if not os.path.isdir(self.data_dir):
            print(f'{self.data_dir} does not exist.')
            raise Exception(f"`{self.data_dir}` does not exist")

    def _download(self):
        pass


    def _process_one_step(self, image, masks, num_paths):
        image = np.asarray(image)
        segmentation_algorithm = slic_fixed(1000, compactness=50, max_iterations=10, sigma=0)
        segmentation = segmentation_algorithm(image)
        adj = segmentation_adjacency(segmentation)
        adj = np.array(adj.todense())
        adj = torch.from_numpy(adj)
        edge_x = Variable(torch.from_numpy(create_edge_list(adj)))
        features = Variable(torch.from_numpy(create_features(image, segmentation))).type(torch.FloatTensor) 
        target_mask = masks
        #y = Variable(torch.from_numpy(create_target(segmentation, target_mask, num_paths)))
        y = Variable(torch.from_numpy(target_mask))
        #num_instance = np.max(target_mask)+1
        data = Data(features, edge_x, y=y, segmentation = Variable(torch.from_numpy(segmentation)))
        return data
    
    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        flag = False
        out_path = osp.join(config.ROOT_PATH,self.split, f'data_{idx}.pt')
        #data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        image, out, num_paths = self.get_groundtruth(idx)
        if num_paths >= config.OUTPUT_LAYER:
            print('Pass: ', idx)
            flag = True
            pass
        else:
            if flag == True:
                idx = idx-1
            data = self._process_one_step(image, out, num_paths)
            torch.save(data, out_path)
            flag = False
        
        # if self.transforms:
        #     out_path = osp.join("/home/rhythm/notebook/vectorData_test/temp/processed/", self.split, f'data_{self.__len__()+idx}.pt')
        #     image, out = self.transforms(image, out)
        #     image = image.permute(1, 2, 0)
        #     data = self._process_one_step(image, out, num_paths)
        #     torch.save(data, out_path)
        #     return data
            return data

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
        svg_name = self.root_dir / self.split / "{:s}.svg".format(self.ids[idx])
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
            # ret , thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
            # contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            mask = np.where(mask == 1, i, 0)  
            #print(mask)
            out += mask
            out = np.asarray(out, dtype=int)
            #print('i: ',i, 'mask: ', out.max())
            out[out>i] = i
        return self.svg_to_png(svg), out, num_paths

