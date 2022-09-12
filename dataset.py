import torch

import cairosvg
import xml.etree.ElementTree as et
import io
from PIL import Image
import numpy as np
from PIL import ImageOps
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from slic_segm import slic_fixed
from utils import segmentation_adjacency, create_edge_list, create_features, create_target

def svg_to_png(svg):
    image = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image)).split()[-1].convert("RGB")
    image = ImageOps.invert(image)
    
    assert (image.size == (64,64))
    return image



def ChineseDataset(data_dir, idx):
        
        split = "train"
        # self.png_name = self.data_dir / "png" / "{:s}.png"
        # self.svg_name = self.data_dir / "svg" / "{:s}.svg"

        with (data_dir / "{:s}.txt".format(split)).open("r") as f:
            ids = [_.strip() for _ in f.readlines()]
        
        
        svg_name = data_dir / "svg" / "{:s}.svg".format(ids[idx])
        
        with svg_name.open('r') as f_svg:
            svg = f_svg.read()


        num_paths = len(et.fromstring(svg)[0])
        target = []
        for i in range(num_paths):
            svg_xml = et.fromstring(svg)
            #svg_xml[0][0] = svg_xml[0][i]
            #del svg_xml[0][1:]
            svg_one = et.tostring(svg_xml, method='xml')

            # leave only one path
            y_png = cairosvg.svg2png(bytestring=svg_one)
            y_img = Image.open(io.BytesIO(y_png))
            mask = (np.array(y_img)[:, :, 3] > 0)
            mask = mask.astype(np.uint8)
            
            image = svg_to_png(svg)
            image = np.asarray(image)

            segmentation_algorithm = slic_fixed(300, compactness=5, max_iterations=20, sigma=0)
            segmentation = segmentation_algorithm(image)

            adj = segmentation_adjacency(segmentation)
            adj = np.array(adj.todense())
            adj = torch.from_numpy(adj).type(torch.FloatTensor)

            edge_x = Variable(torch.from_numpy(create_edge_list(adj))).cuda()
            features = Variable(torch.from_numpy(create_features(image, segmentation))).type(torch.cuda.FloatTensor)

            y = Variable(torch.from_numpy(create_target(segmentation, mask)))
            num_instance = np.max(mask)+1
            target.append(y)

            datasets = []
            datasets.append(Data(features, edge_x, y = target))

            return DataLoader(datasets, batch_size=1), adj, image, segmentation, mask
