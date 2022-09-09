import io
import random
import re
import xml.etree.ElementTree as et
from pathlib import Path
import cv2
import cairosvg
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
import os
from maskrcnn_benchmark.data.transforms.build import build_transforms
from maskrcnn_benchmark.data.transforms.transforms import Compose
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from skimage import color, segmentation

ROOT = "../../data"


def get_dataset(cfg, dataset="ch", subset="train", is_train=True):
    
    assert isinstance(dataset, str)
    assert isinstance(subset, str)
    assert isinstance(is_train, bool)

    # TODO: support multi-class
    # we train a single network on a set of classes (BASEBALL, CAT, CHANDELIER and ELEPHANT), which we call the MULTI-CLASS dataset. We then compute the testing error on a set of unseen classes (BACKPACK and BICYCLE).
    
    datasets_dict = {
        "ch": ChineseDataset,
        "kanji": KanjiDataset,
        "line": LineDataset,
        "cat": QDrawDataset,
        "baseball": QDrawDataset,
        "multi": QDrawDataset,
    }
    
    assert subset in ["train", "test"]
    assert dataset in datasets_dict.keys()
    root = str(Path(ROOT) / dataset)
    
    transforms = build_transforms(cfg, is_train)
    
    return datasets_dict[dataset](root, subset, transforms)


class BaseDataset(torch.utils.data.Dataset):
    
    CLASSES = ("BG", "FG")

    def __init__(self, data_dir, split, transforms=None, size=64, canvas_size=1024):
        
        assert isinstance(data_dir, str)
        assert isinstance(split, str)
        assert isinstance(transforms, Compose)
        assert isinstance(size, int)
        assert isinstance(canvas_size, int)

        self.data_dir = Path("/home/rhythm/notebook/fast-line-drawing-vectorization-master/data/ch/")
        self.split = split
        # self.png_name = self.data_dir / "png" / "{:s}.png"
        # self.svg_name = self.data_dir / "svg" / "{:s}.svg"

        with (self.data_dir / "{:s}.txt".format(split)).open("r") as f:
            self.ids = [_.strip() for _ in f.readlines()]
        
        self.size = size
        self.canvas_size = canvas_size
        self.transforms = transforms
        

    def get_groundtruth(self, idx):
        
        
        raise NotImplementedError

    def __getitem__(self, idx):
        #print(idx)
        #print(self.data_dir)
        
        image, boxlist = self.get_groundtruth(idx)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        return image, boxlist, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        return {"height": self.size, "width": self.size}

    @staticmethod
    def transform_y(y):
        return y

    def _preprocess(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:
            return color.gray2rgb(
                np.reshape(image, (image.shape[0], image.shape[1])))

    def slic(self, image, num_segments, compactness=10, max_iterations=20, sigma=0):
        image = self._preprocess(image)
        return segmentation.slic(image, num_segments, compactness, max_iterations,
                                sigma, start_label=0)

    def slic_fixed(self, num_segments, compactness=1, max_iterations=2, sigma=0):
        def slic_image(self, image):
            return self.slic(image, num_segments, compactness, max_iterations, sigma)

        return slic_image

    def svg_to_png(self, svg):
        image = cairosvg.svg2png(bytestring=svg)
        image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
        image = ImageOps.invert(image)
        assert (image.size == (self.size, self.size))
        
        return image


class ChineseDataset(BaseDataset):
    
    def __init__(self, data_dir, split, transforms=None, size=64, canvas_size=1024):
        
        super().__init__(data_dir, split, transforms, size, canvas_size)
        

    def get_groundtruth(self, idx):

        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()


        num_paths = len(et.fromstring(svg)[0])
        
        for i in range(num_paths):
            svg_xml = et.fromstring(svg)
            #svg_xml[0][0] = svg_xml[0][i]
            #del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)

            #masks.append(mask.astype(np.uint8))

        image = self.svg_to_png(svg)
        image = np.asarray(image)
        segmentation_algorithm = self.slic_fixed(num_segments= 200, compactness=1, max_iterations=20, sigma=0)
        segmentation = segmentation_algorithm(self, image)

        return self.svg_to_png(svg), mask


class KanjiDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=109):
        super().__init__(data_dir, split, transforms, size, canvas_size)

    def get_groundtruth(self, idx):
        bboxes = []
        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        pid = 0
        num_paths = 0
        while pid != -1:
            pid = svg.find('path id', pid + 1)
            num_paths = num_paths + 1
        num_paths = num_paths - 1  # uncount last one

        for i in range(num_paths):
            svg_one = svg
            pid = len(svg_one)

            for j in range(num_paths):
                pid = svg_one.rfind('path id', 0, pid)
                if j != i:
                    id_start = svg_one.rfind('>', 0, pid) + 1
                    id_end = svg_one.find('/>', id_start) + 2
                    svg_one = svg_one[:id_start] + svg_one[id_end:]

            y_png = cairosvg.svg2png(bytestring=svg_one.encode('utf-8'))
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            bboxes.append(self.get_bbox(mask, "xyxy"))
            masks.append(mask.astype(np.uint8))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist


class LineDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=64,
                 canvas_size=64):
        super().__init__(data_dir, split, transforms, size, canvas_size)

    def get_groundtruth(self, idx: int):
        bboxes = []
        masks = []

        svg_name = self.data_dir / "svg" / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        svg_xml = et.fromstring(svg)

        for i in range(len(svg_xml[0])):
            svg_xml = et.fromstring(svg)
            svg_xml[0][0] = svg_xml[0][i]
            del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            assert mask.shape == (self.size, self.size), (self.ids[idx])
            bboxes.append(self.get_bbox(mask, "xyxy"))
            masks.append(mask.astype(np.uint8))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist


class QDrawDataset(BaseDataset):
    def __init__(self, data_dir, split, transforms=None, size=128,
                 canvas_size=128):
        super().__init__(data_dir, split, transforms, size, canvas_size)
        ng_ids = ["4585937717690368", "4583140179836928",  # cat
                  "6231511248404480", "6442058833199104",  # cat
                  "5327861537832960"]  # baseball
        for ng_id in ng_ids:
            if ng_id in self.ids:
                self.ids.remove(ng_id)

    def get_groundtruth(self, idx):
        bboxes = []
        masks = []

        svg_name = self.data_dir / self.split / "{:s}.svg".format(
            self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()

        num_paths = svg.count('polyline')

        for i in range(1, num_paths + 1):
            svg_xml = et.fromstring(svg)
            svg_xml[1] = svg_xml[i]
            del svg_xml[2:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            try:
                bboxes.append(self.get_bbox(mask, "xyxy"))
            except:
                continue
            masks.append(mask.astype(np.uint8))

        # if there is completely no boxes, add dummy boxes and masks
        if len(bboxes) == 0:
            
            dummy_mask = np.zeros(mask.shape, dtype=np.uint8)
            dummy_mask[0:4, 0:4] = 1
            masks.append(dummy_mask.astype(np.uint8))
            bboxes.append(self.get_bbox(dummy_mask, "xyxy"))

        image_size_dict = self.get_img_info(idx)
        image_size = (image_size_dict["height"], image_size_dict["width"])
        boxlist = BoxList(bboxes, image_size, mode="xyxy")
        boxlist.add_field("labels", torch.tensor([1] * len(bboxes)))
        boxlist.add_field("mask", SegmentationMask(masks, image_size, "mask"))
        return self.svg_to_png(svg), boxlist
