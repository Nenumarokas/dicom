from tqdm import tqdm
from array_lib import *
import pydicom as dicom
import numpy as np
import cv2 as cv
import pickle
import os

def read_dicom(input_folder: str) -> np.ndarray:
    files: list[str] = os.listdir(input_folder)
    data = [dicom.dcmread(f'{input_folder}\\{file}') for file in files if file.endswith('.dcm')]
    image = np.array([dicom.pixel_array(datum) for datum in data])
    return image

def normalize_image_colors(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    offsets = (-15, 20)
    seed = (20, 310, 160)
    image_mask = custom_floodfill_3d(image, seed_point=seed, new_value=-1, offsets=offsets)
    return np.where(image_mask, image+50, image)

def within_bounds(x, y):
    return 10 < x < image.shape[1]-10 and 10 < y < image.shape[2]-10

def draw(image: np.ndarray, x: int, y: int, size: int, mark: bool = True):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)).astype(bool)

    for i in range(size):
        for j in range(size):
            if not kernel[i, j]:
                continue
            xi = x - size//2 + i
            yj = y - size//2 + j
            if not within_bounds(xi, yj):
                continue
            image[yj, xi] = mark
    
def draw_annotation(event, x, y, flags, param):
    global drawing, drawing_size, mark
    if not within_bounds(x, y):
        return

    if event == cv.EVENT_MOUSEMOVE:
        global last_x, last_y
        last_x = x
        last_y = y

    match event:
        case cv.EVENT_LBUTTONUP:
            drawing = False
        case cv.EVENT_LBUTTONDOWN:
            drawing = True
            draw(annotated[selected_layer], x, y, drawing_size, mark)
        case cv.EVENT_MOUSEMOVE:
            if drawing:
               draw(annotated[selected_layer], x, y, drawing_size, mark)
        case cv.EVENT_RBUTTONUP:
            mark = not mark
        case _:
            pass

def draw_kernel(image: np.ndarray, x: int, y: int, size: int):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)).astype(bool)

    for i in range(size):
        for j in range(size):
            if within_bounds(i, j):
                image[x+size//2, y+size//2] = kernel[i, j]
    return image

def join_images(image: np.ndarray, annotated: np.ndarray, selected_layer: int, x: int, y: int, size: int):
    shown_image = cv.cvtColor(image[selected_layer], cv.COLOR_GRAY2BGR)
    annotated_shown = shown_image.copy()
    annotated_copy = annotated[selected_layer].copy()
    draw(annotated_copy, x, y, size, mark)
    annotated_shown[annotated_copy] = (0, 0, 255)

    annotated_shown = cv.addWeighted(shown_image, 0.8, annotated_shown, 0.2, 0)
    
    joined = cv.hconcat((annotated_shown, shown_image))
    joined = cv.copyMakeBorder(joined, 0, 100, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    cv.putText(joined, f'{selected_layer+1}/{image.shape[0]}', (400, image.shape[1]+70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv.putText(joined, f'size: {size}', (0, image.shape[1]+70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return joined

def save_annotations(annotation_file: str, annotated: np.ndarray):
    np.save(annotation_file, annotated)

def read_annotations(annotation_file: str):
    return np.load(annotation_file)

if __name__ == '__main__':
    print('\n \n ')
    
    folder_name = '20240923_84'
    annotation_file = f'{folder_name}_annotation.npy'
    
    image = read_dicom(f'{os.getcwd()}\\{folder_name}')
    image = normalize_image_colors(image)
    image = preprocess_image(image)
    
    annotated = np.zeros_like(image).astype(bool)
    if os.path.exists(annotation_file):
        annotated = read_annotations(annotation_file)

    selected_layer = 20
    drawing_size = 25
    mark = True
    last_x = 100
    last_y = 100
    
    cv.namedWindow('image')
    drawing = False
    cv.setMouseCallback('image', draw_annotation)

    while True:
        joined = join_images(image, annotated, selected_layer, last_x, last_y, drawing_size)
        cv.imshow('image', joined)
        
        key = cv.waitKey(1)
        match key:
            case 113:# q
                break
            case 100:# d
                if selected_layer > 0:
                    selected_layer -= 1
            case 102:# f
                if selected_layer < image.shape[0] - 1:
                    selected_layer += 1
            case 103:# g
                if selected_layer < image.shape[0] - 11:
                    selected_layer += 10
                else:
                    selected_layer = image.shape[0] - 1
            case 115:# s
                if selected_layer > 10:
                    selected_layer -= 10
                else:
                    selected_layer = 0
            case 114:# r
                annotated[selected_layer][:] = False
            case 120:# x
                if drawing_size < 51:
                    drawing_size += 2
            case 122:# z
                if drawing_size > 4:
                    drawing_size -= 2
            case _:
                continue
    
    save_annotations(annotation_file, annotated)
    