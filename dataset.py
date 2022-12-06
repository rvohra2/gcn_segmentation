import torch
import os
import cairosvg
import xml.etree.ElementTree as et
import io
from PIL import Image
import numpy as np
from PIL import ImageOps
from pathlib import Path
from utils import build_transforms, segmentation_adjacency, create_edge_list, create_features, create_target, select_mask_color, Stroke, getSkeleton,showImage, generate_strokes2, draw_strokes, generate_final_strokes
import sys
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset, Data
from slic_segm import slic_fixed
from torch.autograd import Variable
import os.path as osp
import config
import cv2
from skimage.morphology import skeletonize


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (80, 127, 255), (255, 0, 255),
          (255, 255, 0), (96, 164, 244)]
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
        segmentation_algorithm = slic_fixed(300, compactness=10, max_iterations=10, sigma=0)
        segmentation = segmentation_algorithm(image)
        adj = segmentation_adjacency(segmentation)
        adj = np.array(adj.todense())
        adj = torch.from_numpy(adj)
        edge_x = Variable(torch.from_numpy(create_edge_list(adj)))
        features = Variable(torch.from_numpy(create_features(image, segmentation))).type(torch.FloatTensor) 
        target_mask = masks
        y = Variable(torch.from_numpy(create_target(segmentation, target_mask, num_paths)))
        #y = Variable(torch.from_numpy(target_mask))
        #num_instance = np.max(target_mask)+1
        data = Data(features, edge_x, y=y, segmentation = Variable(torch.from_numpy(segmentation)))
        return data
    
    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        
        out_path = osp.join(config.ROOT_PATH,self.split, f'data_{idx}.pt')
        #data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        image, out, num_paths = self.get_groundtruth(idx)
        
        data = self._process_one_step(image, out, num_paths=config.OUTPUT_LAYER)
        torch.save(data, out_path)
            
        
        # if self.transforms:
        #     out_path = osp.join("/home/rhythm/notebook/vectorData_test/temp/processed/", self.split, f'data_{self.__len__()+idx}.pt')
        #     image, out = self.transforms(image, out)
        #     image = image.permute(1, 2, 0)
        #     data = self._process_one_step(image, out, num_paths)
        #     torch.save(data, out_path)
        #     return data
            #return data

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
        svg_name = self.root_dir / self.split / "{:s}.svg".format(self.ids[idx])
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()
        num_paths = svg.count('polyline')
       
        
        #print(num_paths)
        out = np.zeros(shape=(128,128,3))
        out1 = np.zeros(shape=(128,128,3))
        #out1 = np.asarray(out1, dtype=int)

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
            #mask = (np.array(y_img)[:, :, 3] > 0).astype(np.uint8)
            #y_img1 = (np.array(y_img)).astype(np.uint8)
            y_img1 = np.array(y_img.convert('RGB'))
            
            mask = cv2.cvtColor(y_img1, cv2.COLOR_RGB2GRAY)
            ret , thrash = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
            # kernel = np.ones((2,2),np.uint8)
            # thrash = cv2.erode(thrash,kernel,iterations = 1)
            thrash1 = skeletonize(thrash).astype(np.uint8)
            # plt.imshow(thrash1, cmap='gray')
            # plt.show()

            if thrash1.max() == 0:
                continue
            else:
                
                contours,hierarchy =  cv2.findContours(thrash1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                
                hierarchy = hierarchy[0]
                for i, c in enumerate(contours):
                    if hierarchy[i][2] < 0 and hierarchy[i][3] < 0:
                        # approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, False), False)
                        # print(len(approx))
                        # dst = cv2.Canny(y_img1, 1., 15., 3)
                        # lines = cv2.HoughLinesP(dst, 1., np.pi/180., 15, 10, 100.)
                        # if lines is not None:
                        #     print('Line detected')
                        #     for this_line in lines:
                        #         cv2.line(y_img1,
                        #                 (this_line[0][0], this_line[0][1]),
                        #                 (this_line[0][2], this_line[0][3]),
                        #                 [0, 0, 255], 3, 8)
                        if cv2.arcLength(c, False) < 19:
                            out[:,:,0] = thrash * 255 # for red
                            out[:,:,1] = thrash * 0 # for green
                            out[:,:,2] = thrash * 255 # for blue
                        else:
                            approx = cv2.approxPolyDP(c, 0.07* cv2.arcLength(c, False), False)
                            if len(approx) >4:
                                out[:,:,0] = thrash * 0 # for red
                                out[:,:,1] = thrash * 0 # for green
                                out[:,:,2] = thrash * 255 # for blue
                            else:
                                out[:,:,0] = thrash * 255 # for red
                                out[:,:,1] = thrash * 0 # for green
                                out[:,:,2] = thrash * 0 # for blue

                    else:

                        if (cv2.contourArea(c) > cv2.arcLength(c, True)) == True:
                            out[:,:,0] = thrash * 0 # for red
                            out[:,:,1] = thrash * 255 # for green
                            out[:,:,2] = thrash * 0 # for blue
                            #out[(thrash == 1).all()] = [0,255,0]
                            #cv2.drawContours(y_img1, contours, i, (0, 255, 0), 2)
                            
                        else:
                            approx = cv2.approxPolyDP(c, 0.07 * cv2.arcLength(c, False), False)
                            if len(approx) >4:
                                out[:,:,0] = thrash * 0 # for red
                                out[:,:,1] = thrash * 0 # for green
                                out[:,:,2] = thrash * 255 # for blue
                            else:
                                out[:,:,0] = thrash * 255 # for red
                                out[:,:,1] = thrash * 0 # for green
                                out[:,:,2] = thrash * 0 # for blue
                out = out.astype(np.uint8)
                out1 += out
                out1 = np.asarray(out1, dtype=int)
                #out1[out1>255] = 255
                plt.imshow(out1)
                plt.show()
        
        r = np.array([0, 255, 0])
        g = np.array([0, 0, 255])
        b = np.array([255, 0, 0])
        x = np.array([255, 0, 255])
        label_seg = np.zeros((128,128), dtype=np.int)
        label_seg[(out1==r).all(axis=2)] = 1
        label_seg[(out1==g).all(axis=2)] = 2
        label_seg[(out1==b).all(axis=2)] = 3
        label_seg[(out1==x).all(axis=2)] = 4
        # cond = (out1!=r and out1!=g and out1!=b and out1!=x)
        # label_seg[(cond == True).all(axis=2)] = 5
        # plt.imshow(label_seg)
        # plt.show()


        # mask = np.where(mask == 1, i, 0)  
        # #print(mask)
        # out += mask
        # out = np.asarray(out, dtype=int)
        # #print('i: ',i, 'mask: ', out.max())
        # out[out>i] = i
        return self.svg_to_png(svg), label_seg, num_paths


# num_labels, labels = cv2.connectedComponents(thrash)
    
# # Map component labels to hue val, 0-179 is the hue range in OpenCV
# label_hue = np.uint8(179*labels/np.max(labels))
# blank_ch = 255*np.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# # Converting cvt to BGR
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# # set bg label to black
# labeled_img[label_hue==0] = 0
# print('Unique Paths: ', len(np.unique(labeled_img)))
# #Showing Image after Component Labeling
# plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title("Image after Component Labeling")
# plt.show()