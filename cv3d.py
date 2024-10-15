from tqdm import tqdm
import scipy.ndimage as nd
import numpy as np
import cv2 as cv
import time
import os

def dilate(image: np.ndarray, iterations: int = 1, border_value: int = 0) -> np.ndarray:
    dilated = nd.binary_dilation(image, iterations=iterations, border_value=border_value)
    return dilated

def erode(image: np.ndarray, iterations: int = 1, border_value: int = 1) -> np.ndarray:
    eroded = nd.binary_erosion(image, iterations=iterations, border_value=border_value)
    return eroded

def create_cross_kernel() -> np.ndarray:
    kernel = np.zeros((3, 3, 3), dtype=int)
    kernel[1, :, 1] = 1
    kernel[:, 1, 1] = 1
    kernel[1, 1, :] = 1
    return kernel

def make_skeleton(image: np.ndarray) -> np.ndarray:
    timer = time.time()
    image = image.astype(bool)
    mask = image.copy()

    skeleton = np.zeros_like(image)
    counter = 0
    while np.count_nonzero(image) > 0 and counter < 20:
        eroded = nd.binary_erosion(image, border_value=1, mask=mask)
        temp = nd.binary_dilation(eroded, border_value=0, mask=mask)
        temp = np.bitwise_xor(image, temp)
        skeleton = np.logical_or(skeleton, temp)
        image = eroded
        
        counter += 1
        taken = time.time() - timer
        print(np.sum(skeleton.astype(int)), end = " ")
        print(f'----skeletonizing: {round(taken, 3)}s for {counter}its ({round(taken/counter, 3)} per it)', end='\r')
    return skeleton

if __name__ == '__main__':
    size = 322
    matrix = np.zeros((size, size, size))
    matrix[:3,:3,:3] = 1
    matrix[1:4,1:4,1:4] = 1
    matrix[2:5,2:5,2:5] = 1
    # print(matrix)

    timer = time.time()

    # for i in range(50000):
    m = make_skeleton(matrix)
    print(f'creating skeleton:{round(time.time() - timer, 3)}s')
    timer = time.time()

    print(m[:5,:5,:5])
