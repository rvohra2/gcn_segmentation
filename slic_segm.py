from skimage import color, segmentation
import numpy as np

def _preprocess(image):
    if len(image.shape) == 2 or image.shape[2] == 1:
        return color.gray2rgb(
            np.reshape(image, (image.shape[0], image.shape[1])))
    else:
        return image

def slic(image, num_segments, compactness=10, max_iterations=20, sigma=0):
    image = _preprocess(image)
    return segmentation.slic(image, num_segments, compactness, max_iterations,
                             sigma, start_label=0)

def slic_fixed(num_segments, compactness=1, max_iterations=2, sigma=0):
    def slic_image(image):
        return slic(image, num_segments, compactness, max_iterations, sigma)

    return slic_image