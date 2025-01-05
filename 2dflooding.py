#!/usr/bin/env python

from ply_creation_lib import create_ply
from tqdm import tqdm
from array_lib import *
import pydicom as dicom
import numpy as np
import cv2 as cv
import os

def read_dicom(input_folder: str) -> np.ndarray:
    files: list[str] = os.listdir(input_folder)
    data = [dicom.dcmread(f'{input_folder}\\{file}') for file in files if file.endswith('.dcm')]
    image = np.array([dicom.pixel_array(datum) for datum in data])
    return image

def normalize_image_colors(image: np.ndarray, minmax: tuple[int, int]) -> np.ndarray:
    image = image.copy()
    min_val, max_val = minmax
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def get_average_color(image: np.ndarray, seed: tuple[int, int], lower_diff: int) -> np.ndarray:
    x, y = seed
    lower_y = max(y-25, 0)
    upper_y = min(y+25, image.shape[0])
    lower_x = max(x-25, 0)
    upper_x = min(x+25, image.shape[1])
    start_value = int(np.average(image[lower_y:upper_y, lower_x:upper_x]))
    _, thresh = cv.threshold(image, start_value - lower_diff, 100, cv.THRESH_BINARY)
    _, flooded, _, _ = cv.floodFill(thresh, None, seed, 255, lower_diff, 100)
    _, thresh = cv.threshold(flooded, 250, 255, cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours, start_value - lower_diff

def nothing(x):
    pass

def clip_colors(minmax: tuple[int, int], image: np.ndarray, thresh: int) -> np.ndarray:
    image = image.copy()
    
    min_val, max_val = minmax
    image = np.clip(image, 0, max_val)
    min_val = 0
    image = np.where(image > thresh, image, 0)
    image = np.where(image > 2000, 0, image)
    max_val = 2000
    image = (image - min_val) / (max_val - min_val) * 200
    image = np.where(image > 0, image + 55, 0)
    # image = np.clip(image, 0, 255)
    # image = np.where(image > 255, 0, image)    
    return np.where(image > 0, 100, 0).astype(np.uint8)

if __name__ == '__main__':
    print('\n \n ')
    
    folder = f'{os.getcwd()}\\{'20240923'}'
    image: np.ndarray = read_dicom(folder)
    minmax = (np.min(image), np.max(image))
    
    selected_layer = 20
    initial_threshold = 200
    layer = 0


    image = np.array([clip_colors(minmax, l, initial_threshold) for l in image])
    aorta_seed = find_aorta(image[selected_layer])
    flooded = floodfill_3d(image, 255, seed=(selected_layer, *(aorta_seed[::-1])))
    filtered = np.where(flooded > 200, 255, 0).astype(np.uint8)

    heartless = filter_horizontal_slices(filtered, selected_layer)

    create_ply(heartless, 'heartless.ply')
    create_ply(filtered, 'filtered.ply')
