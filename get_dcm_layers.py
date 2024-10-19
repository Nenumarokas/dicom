#!/usr/bin/env python

from tqdm import tqdm
import pydicom as dicom
import numpy as np
import cv2 as cv
import struct
import time
import os

print('\n \n ')

def normalize_image(image: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
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
    
    folder = f'{os.getcwd()}\\20240506'

    files = os.listdir(folder)
    vertex_count = 0
    
    images = []
    
    for file in tqdm(files, ncols=50):
        data = dicom.dcmread(f'{folder}\\{file}')
        image = dicom.pixel_array(data)
        images.append(image)
        
    min_value = min(np.min(image) for image in images)
    max_value = max(np.max(image) for image in images)
    images = [normalize_image(image, min_value, max_value) for image in images]
    
    cv.namedWindow('image')
    cv.createTrackbar('layer', 'image', 0, len(images)-1, nothing)
    cv.createTrackbar('lower_threshold', 'image', 0, 255, nothing)
    cv.createTrackbar('top_part', 'image', 0, 512, nothing)
    cv.createTrackbar('bottom_part', 'image', 512, 512, nothing)
    cv.createTrackbar('left_part', 'image', 0, 512, nothing)
    cv.createTrackbar('right_part', 'image', 512, 512, nothing)
        
    while True:
        selected_layer = cv.getTrackbarPos('layer', 'image')
        lower_threshold = cv.getTrackbarPos('lower_threshold', 'image')
        top_part = cv.getTrackbarPos('top_part', 'image')
        bottom_part = cv.getTrackbarPos('bottom_part', 'image')
        left_part = cv.getTrackbarPos('left_part', 'image')
        right_part = cv.getTrackbarPos('right_part', 'image')
        
        
        image = images[selected_layer].copy()
        image = cv.flip(image, 0)
        image[:top_part,:] = 0
        image[:,:left_part] = 0
        image[:,right_part:] = 0
        image[bottom_part:,:] = 0
        print(f'{top_part}-{bottom_part} x {left_part}-{right_part}')
        _, image = cv.threshold(image, lower_threshold, 255, cv.THRESH_TOZERO)
        
        # _, image = cv.threshold(image, upper_threshold, 255, cv.THRESH_TOZERO_INV)
        
        cv.imshow('image', image)
        if cv.waitKey(1) == ord('q'):
            break
