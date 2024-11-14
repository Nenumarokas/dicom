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
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def nothing(x):
    pass

if __name__ == '__main__':
    folder = f'{os.getcwd()}\\20240923'

    image = read_dicom(folder)
    image = normalize_image_colors(image)
    
    cv.namedWindow('image')
    cv.createTrackbar('layer', 'image', 0, len(image)-1, nothing)
    cv.createTrackbar('lower_threshold', 'image', 0, 255, nothing)
    while True:
        selected_layer = cv.getTrackbarPos('layer', 'image')
        lower_threshold = cv.getTrackbarPos('lower_threshold', 'image')
       
        selected_image = image[selected_layer].copy()
        _, selected_image = cv.threshold(selected_image, lower_threshold, 255, cv.THRESH_TOZERO)
        
        cv.imshow('image', selected_image)
        if cv.waitKey(1) == ord('q'):
            break
