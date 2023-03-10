import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import cairosvg
from PIL import Image
import io
from PIL import ImageOps
import xml.etree.ElementTree as et

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(a)
a[a<5] = 4
print('a1: ', a)




# Load image, grayscale, Gaussian blur, Otsu's threshold, dilate

def svg_to_png(svg):
    
    image = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image)).split()[-1].convert("L")
    image = ImageOps.invert(image)
    image.thumbnail((128,128), Image.ANTIALIAS)
    #assert (image.size == (32,32))
    return image

def get_bbox(mask, mode="xyxy"):
        assert isinstance(mask, np.ndarray)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if mode == "xyxy":
            
            return cmin, rmin, cmax, rmax
        else:
            raise NotImplementedError

svg_name = os.path.join('/home/rhythm/notebook/vectorData_test/svg/1.svg')
with open(svg_name, 'r') as f_svg:
    svg = f_svg.read()

image = svg_to_png(svg)
##Uncomment for line drawing dataset
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
    mask = mask.astype(np.uint8)
    


    original = image.copy()
    plt.imshow(mask, cmap='gray')
    plt.show()

    image = np.asarray(image)
    original = np.asarray(original)


    # Find contours, obtain bounding box coordinates, and extract ROI
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0
    for c in cnts:
        x, y, w, h = get_bbox(mask, "xyxy")
        cv2.rectangle(mask, (x, y), (x + w, y + h), (36,255,12), 2)
        ROI = original[y:y+h, x:x+w]
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        image_number += 1

cv2.imshow('image', image)
