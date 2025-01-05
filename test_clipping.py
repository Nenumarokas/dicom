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

def show_cursor_location(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        single_image = images[x//512][layer].copy()
        normalized = normalize_image_colors(single_image, minmax[x//512])
        cv.circle(normalized, (x%512, y), 3, 0, -1)
        cv.putText(normalized, str(single_image[y, x%512]), (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, 255, 2)
        
        cv.imshow('smaller_image', normalized)

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
    
    
    return image.astype(np.uint8)

if __name__ == '__main__':
    print('\n \n ')
    
    folders = ['20240923', '20241209_17', '20241209_46']
    folders = [f'{os.getcwd()}\\{f}' for f in folders]
    images: list[np.ndarray] = [read_dicom(f) for f in folders]
    minmax = [(np.min(image), np.max(image)) for image in images]
    
    min_height = min(image.shape[0] for image in images)
    selected_layer = 20
    layer = 0
    
    cv.namedWindow('image')
    cv.createTrackbar('threshold', 'image', 200, 2000, nothing)
    cv.createTrackbar('layer', 'image', selected_layer, min_height-1, nothing)
    cv.setMouseCallback('image', show_cursor_location)
    while True:
        threshold = cv.getTrackbarPos('threshold', 'image')
        layer = cv.getTrackbarPos('layer', 'image')

        selected_images = []
        for i, image in enumerate(images):
            selected_images.append(clip_colors(minmax[i], image[layer], threshold))
        selected_image = cv.hconcat(selected_images)
        
        cv.imshow('image', selected_image)
        if cv.waitKey(1) == ord('q'):
            break

