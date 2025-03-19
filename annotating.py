from tqdm import tqdm
from array_lib import *
from ply_creation_lib import create_ply
import pydicom as dicom
import nibabel as nib
import numpy as np
import cv2 as cv
import pickle
import copy
import os

def read_nii(nii_path: str) -> np.ndarray:
    nii_img = nib.load(nii_path)  # Load NIfTI file
    numpy_array = nii_img.get_fdata()  # Get image data as NumPy array
    return numpy_array

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

def clip_image_colors(image: np.ndarray, min_val: int, max_val: int) -> np.ndarray:
    image = copy.deepcopy(image)
    image = np.where((image > min_val) | (image < max_val), image, 0)
    image = (image - min_val) / (max_val - min_val) * 200
    image = np.where(image > 0, image + 55, 0)
    return image.astype(np.uint8)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    offsets = (-15, 20)
    seed = (20, 310, 160)
    image_mask = custom_floodfill_3d(image, seed_point=seed, new_value=-1, offsets=offsets)
    return np.where(image_mask, image+50, image)

def within_bounds(x, y):
    return 10 < x < image.shape[1] - 10 and 10 < y < image.shape[2] - 10

def draw(original_image: np.ndarray, annotation_image: np.ndarray, x: int, y: int, size: int, mark: bool, show: bool, flood: bool = False):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size)).astype(bool)

    if flood:
        if original_image[x, y] == 0:
            return
        
        if show:
            copied = original_image.copy()
            _, _, b, _ = cv.floodFill(copied, None, (x, y), 255, 20, 20)
            b: np.ndarray = b[1:-1, 1:-1].astype(bool)
            annotation_image[b] = ~annotation_image[b]
        else:
            annotation_image[y-3:y+2, x-3:x+2] = True
        return
    
    for i in range(size):
        for j in range(size):
            if not kernel[i, j]:
                continue
            xi = x - size//2 + i
            yj = y - size//2 + j
            if not within_bounds(xi, yj):
                continue
            annotation_image[yj, xi] = (mark==1)
    
def draw_annotation(event, x, y, flags, param):
    x = x//2
    y = y//2
    global drawing, drawing_size, mark, floodfill
    if not within_bounds(x, y):
        return

    if event == cv.EVENT_MOUSEMOVE:
        global last_x, last_y
        last_x = x
        last_y = y

    # flood = flags & cv.EVENT_FLAG_CTRLKEY
    match event:
        case cv.EVENT_LBUTTONUP:
            drawing = False
        case cv.EVENT_LBUTTONDOWN:
            drawing = True
            draw(image[selected_layer], annotated[selected_layer], x, y, drawing_size, mark, show=True, flood=floodfill)
        case cv.EVENT_MOUSEMOVE:
            if drawing:
               draw(image[selected_layer], annotated[selected_layer], x, y, drawing_size, mark, show=True, flood=floodfill)
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

def join_images(image: np.ndarray, true_image: np.ndarray, annotated: np.ndarray, selected_layer: int, mark: bool, x: int, y: int, size: int, floodfill: bool):
    shown_image = cv.cvtColor(image[selected_layer], cv.COLOR_GRAY2BGR)
    true_slice = cv.cvtColor(true_image[selected_layer], cv.COLOR_GRAY2BGR)
    annotated_shown = shown_image.copy()
    annotated_copy = annotated[selected_layer].copy()
    draw(image[selected_layer], annotated_copy, x, y, size, mark, show=False)
    annotated_shown[annotated_copy] = (0, 0, 255)

    annotated_shown = cv.addWeighted(shown_image, 0.5, annotated_shown, 0.5, 0)
    true_slice = cv.addWeighted(true_slice, 0.8, annotated_shown, 0.2, 0)
    
    joined = cv.hconcat((annotated_shown, true_slice))
    cv.putText(joined, f'{selected_layer+1}/{image.shape[0]}', (image.shape[0]*3//4, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    mark_sign = '+' if mark else '-'
    flood_sign = 'F' if floodfill else ''
    cv.putText(joined, f'{size}{mark_sign}{flood_sign}', (image.shape[0]*3//5, image.shape[1]-20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv.putText(joined, f'{size}', (image.shape[0]*3//5, image.shape[1]-20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    return joined

def save_annotations(annotation_file: str, annotated: np.ndarray):
    np.save(annotation_file, annotated)

def read_annotations(annotation_file: str):
    return np.load(annotation_file)

if __name__ == '__main__':
    timer = time.time()
    
    dcm_folder = '20250224_48'
    nii_file = '1.img.nii.gz'
    dcm_mode = False

    min_val = 50
    max_val = 2000
    if dcm_mode:
        annotation_file = f'{dcm_folder}_annotation.npy'
        image = read_dicom(f'{os.getcwd()}\\{dcm_folder}')
    else:
        annotation_file = f'{nii_file}_annotation.npy'
        image = read_nii(f'{os.getcwd()}\\dataset\\all\\{nii_file}')
        image = np.transpose(image, (2, 1, 0))

    print(f'reading: {round(time.time() - timer, 2)}s')
    timer = time.time()

    true_image = normalize_image_colors(image)
    print(f'normalizing: {round(time.time() - timer, 2)}s')
    timer = time.time()

    image = clip_image_colors(image, min_val, max_val)
    print(f'clipping: {round(time.time() - timer, 2)}s')
    timer = time.time()
    
    # image = preprocess_image(image)
    
    annotated = np.zeros_like(image).astype(bool)
    if os.path.exists(annotation_file):
        annotated = read_annotations(annotation_file)

    print(f'annotations: {round(time.time() - timer, 2)}s')
    timer = time.time()

    selected_layer = 20
    drawing_size = 9
    floodfill = False
    mark = True
    last_x = 100
    last_y = 100
    
    cv.namedWindow('image')
    drawing = False
    cv.setMouseCallback('image', draw_annotation)

    print(f'setting up: {round(time.time() - timer, 2)}s')
    timer = time.time()

    while True:
        joined = join_images(image, true_image, annotated, selected_layer, mark, last_x, last_y, drawing_size, floodfill)
        
        joined = cv.resize(joined, (joined.shape[1]*2, joined.shape[0]*2))
        cv.imshow('image', joined)
        
        key = cv.waitKey(1)
        match key:
            case 97:# a
                floodfill = not floodfill
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
            case 113:# q
                break
            case 114:# r
                annotated[selected_layer][:] = False
            case 115:# s
                if selected_layer > 10:
                    selected_layer -= 10
                else:
                    selected_layer = 0
            case 97:# a
                floodfill = not floodfill
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
    
    if np.sum(annotated.astype(int)) < 1000:
        exit()
    
    save_annotations(annotation_file, annotated)
    create_ply(annotated, 'annotations.ply')