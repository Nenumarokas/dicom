#!/usr/bin/env python

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

def normalize_image_colors(image: np.ndarray) -> np.ndarray:
    # min_val, max_val = get_hist(image)
    # mask = image < 0
    image = np.clip(image, 0, 9999)
    min_val = np.min(image)
    max_val = -min_val + 300
    image = (image - min_val) / (max_val - min_val) * 200
    image = np.clip(image, 0, 255)
    image = np.where(image > 0, image + 55, 0)
    image = np.where(image > 255, 0, image)
    return image.astype(np.uint8)

def clip_colors(image: np.ndarray) -> np.ndarray:
    min_val = 0
    image = np.where((image > 250) | (image < 2000), image, 0)
    max_val = 2000
    image = (image - min_val) / (max_val - min_val) * 200
    image = np.where(image > 0, image + 55, 0)
    
    return image.astype(np.uint8)

def top_layer_id(image: np.ndarray) -> int:
    last_layer = 0
    for i, layer in enumerate(image[:20]):
        if i < 10:
            continue
        _, thresholded = cv.threshold(layer, 1, 255, cv.THRESH_BINARY)
        pixel_sum = np.sum(thresholded)
        if last_layer == pixel_sum:
            return i
        last_layer = pixel_sum

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

if __name__ == '__main__':
    print('\n \n ')
    
    folder = f'{os.getcwd()}\\20241209_46' #17/46
    # folder = f'{os.getcwd()}\\20240923'
    image = read_dicom(folder)
    
    # image = normalize_image_colors(image)
    image = clip_colors(image)
    
    selected_layer = 20
    
    cv.namedWindow('image')
    cv.resizeWindow('image', 512, 512)
    cv.createTrackbar('left_side', 'image', 100, 511, nothing)
    cv.createTrackbar('top_side', 'image', 100, 511, nothing)
    cv.createTrackbar('lower_offset', 'image', 10, 100, nothing)
    cv.createTrackbar('layer', 'image', selected_layer, len(image)-1, nothing)
    
    mask_thresh = 0
    while True:
        left_side = cv.getTrackbarPos('left_side', 'image')
        top_side = cv.getTrackbarPos('top_side', 'image')
        lower_offset = cv.getTrackbarPos('lower_offset', 'image')
        layer = cv.getTrackbarPos('layer', 'image')
       
        selected_image = image[layer].copy()
        mask, thresh_value = get_average_color(selected_image, (left_side, top_side), lower_offset)
        if cv.waitKey(1) == ord('s'):
            mask_thresh = thresh_value
        _, selected_image = cv.threshold(selected_image, mask_thresh, 255, cv.THRESH_TOZERO)
        selected_image = cv.cvtColor(selected_image, cv.COLOR_GRAY2BGR)
        
        
        cv.drawContours(selected_image, mask, -1, (0, 0, 255), 1)
        cv.line(selected_image, (left_side, 0), (left_side, 512), 255, 1)
        cv.line(selected_image, (0, top_side), (512, top_side), 255, 1)
        
        selected_image = cv.flip(selected_image, 0)
        
        cv.imshow('image', selected_image)
        if cv.waitKey(1) == ord('q'):
            print(f'offsets = ({-lower_offset}, {20})')
            print(f'seed = ({selected_layer}, {top_side}, {left_side})')
            break
