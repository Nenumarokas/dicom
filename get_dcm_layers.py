#!/usr/bin/env python

from tqdm import tqdm
import pydicom as dicom
import numpy as np
import cv2 as cv
import struct
import time
import os

print('\n \n ')

def normalize_image(image: np.ndarray) -> np.ndarray:
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def nothing(x):
    pass

def remove_largest_contour(image: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    
    image_area = image.shape[0] * image.shape[1]
    largest_area = 0
    
    largest_contour = contours[0]
    for contour in contours:
        area = cv.contourArea(contour)
        if area > largest_area and area > image_area * 0.05:
            largest_area = area
            largest_contour = contour
    mask = np.zeros_like(image)
    cv.drawContours(mask, [largest_contour], -1, 255, -1)
    return image & ~mask

if __name__ == '__main__':
    """
    ideas:
    - could also use kmeans in 3D space to distinguish different colored materials
    - can floodfill out the irrelevant stuff
    """
    
    folder = f'{os.getcwd()}\\Bandymas_1_20240923'

    files = os.listdir(folder)
    vertex_count = 0
    
    images = []
    for file in tqdm(files, ncols=50):
        data = dicom.dcmread(f'{folder}\\{file}')
        # image_position = data.get((0x0020, 0x0032))
        # image_orientation = data.get((0x0020, 0x0037))
        # pixel_spacing = data.get((0x0028, 0x0030))
        # print(pixel_spacing)
        
        # print(image_position)
        # print(image_orientation)
        # print(image_height)
        # print()
        
        image = normalize_image(dicom.pixel_array(data))
        images.append(image)
        
    cv.namedWindow('image')
    cv.createTrackbar('layer', 'image', 0, len(images)-1, nothing)
    cv.createTrackbar('threshold', 'image', 0, 255, nothing)
        
    while True:
        selected_layer = cv.getTrackbarPos('layer', 'image')
        selected_threshold = cv.getTrackbarPos('threshold', 'image')
        
        image = images[selected_layer]
        _, image = cv.threshold(image, selected_threshold, 255, cv.THRESH_TOZERO)
        # image = remove_largest_contour(image)
        
        
        
        cv.imshow('image', image)
        if cv.waitKey(1) == ord('q'):
            break
