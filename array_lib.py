from tqdm import tqdm
from numba import njit
import scipy.ndimage as nd
import numpy as np
import cv2 as cv
import time
import os

def dilate_3d(image: np.ndarray, mask: np.ndarray = None, kernel_size: int = 0, iterations: int = 1, border_value: int = 0) -> np.ndarray:
    kernel = None
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return nd.binary_dilation(image, mask=mask, structure=kernel, iterations=iterations, border_value=border_value)

def erode_3d(image: np.ndarray, mask: np.ndarray = None, kernel_size: int = 0, iterations: int = 1, border_value: int = 1) -> np.ndarray:
    kernel = None
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return nd.binary_erosion(image, mask=mask, structure=kernel, iterations=iterations, border_value=border_value)    

def floodfill_3d(image: np.ndarray, new_value: int = 2, seed: tuple = (0, 0, 0)) -> np.ndarray:
    if image.dtype == bool:
        raise ValueError('\"image\" parameter cannot be of boolean type. Please use the floodfill_3d_mask function')
    
    structure = np.ones((3, 3, 3), dtype=int)
    flood_filled, _ = nd.label(image == image[seed], structure=structure)
    image[flood_filled == flood_filled[seed]] = new_value
    return image

def floodfill_3d_mask(mask: np.ndarray, seed: tuple = (0, 0, 0)) -> np.ndarray:
    if mask.dtype != bool:
        raise ValueError('\"mask\" parameter must be of boolean type.')
    
    structure = np.ones((3, 3, 3), dtype=int)
    image = mask.astype(int)
    
    flood_filled, _ = nd.label(image == image[seed], structure=structure)
    image[flood_filled == flood_filled[seed]] = 2
    return image > 1

def threshold_3d(image: np.ndarray, threshold: int, new_value: int, threshold_type: int):
    if threshold_type == cv.THRESH_BINARY:
        return np.where(image > threshold, image, new_value)
    if threshold_type == cv.THRESH_BINARY_INV:
        return np.where(image < threshold, image, new_value)
    return image

def crop_array(image: np.ndarray, padding: int = 0):
    non_zero_indices = np.argwhere(image)
    min_coords = np.maximum(non_zero_indices.min(axis=0) - padding, 0)
    max_coords = np.minimum(non_zero_indices.max(axis=0) + padding, image.shape)
    cropped_matrix = image[min_coords[0]:max_coords[0]+1, 
                            min_coords[1]:max_coords[1]+1, 
                            min_coords[2]:max_coords[2]+1]
    return cropped_matrix

def remove_heart(mask: np.ndarray):
    if (mask.dtype != bool):
        raise ValueError('\"mask\" parameter must be of boolean type.')

    timer = time.time()
    
    eroded = erode_3d(mask, kernel_size=3, iterations=10)
    print(f'\t--eroding: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    dilated = dilate_3d(eroded, kernel_size=3, iterations=12)
    print(f'\t--dilating: {round(time.time() - timer, 3)}s')
    timer = time.time()

    return mask & ~dilated

def distinguish_3d(mask: np.ndarray):
    if (mask.dtype != bool):
        raise ValueError('\"mask\" parameter must be of boolean type.')
    
    timer = time.time()
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = nd.label(mask, structure=structure)
    print(f'\t--labeling: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    
    print(f'{num_features} features')
    empty = np.zeros_like(mask).astype(bool)
    
    counter = 0
    for label_num in range(1, num_features + 1):
        flat_indices = np.flatnonzero(labeled_array == label_num)
        blob_indices = np.unravel_index(flat_indices, labeled_array.shape)
        blob_coords = np.column_stack(blob_indices)
        
        if len(blob_coords) < 500: continue
        if coordinate_bounds_3d(blob_coords) < 1e5: continue
        if len(blob_coords) > 100000: continue
        
        counter += 1
        empty[blob_coords[:, 0], blob_coords[:, 1], blob_coords[:, 2]] = True
    
    print(f'\t--dividing blobs: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    return empty

def fill_holes(data: np.ndarray, kernel_size: int):
    timer = time.time()
    
    mask = (data > 0).astype(int)
    print(np.unique(mask))
    print(f'\t--masking: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    result = nd.convolve(mask, kernel, mode='constant', cval=0)
    
    print(np.unique(result))
    print(f'\t--convolving: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    return np.where(result > 10, 255, 0)

def filter_points(data: np.ndarray):
    timer = time.time()
    
    mask = data > 0
    print(f'\t--masking: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    dilated = dilate_3d(mask, iterations=5)
    print(f'\t--dilating: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    flooded = floodfill_3d(dilated.astype(int), new_value=255)
    flooded = np.where(flooded < 255, 255, 0)
    print(f'\t--flooding: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    eroded = erode_3d(flooded, iterations=5)
    print(f'\t--eroding: {round(time.time() - timer, 3)}s')
    timer = time.time()
    return eroded

@njit
def extract_blob_indices(labeled_array: np.ndarray, label_num: int):
    coords = []
    for i in range(labeled_array.shape[0]):
        for j in range(labeled_array.shape[1]):
            for k in range(labeled_array.shape[2]):
                if labeled_array[i, j, k] == label_num:
                    coords.append((i, j, k))
    return coords

@njit
def coordinate_bounds_3d(blob_coords: np.ndarray):
    ans = 1
    for i in range(3):
        min_val = blob_coords[0, i]
        max_val = blob_coords[0, i]
        for coord in blob_coords:
            if coord[i] < min_val:
                min_val = coord[i]
            if coord[i] > max_val:
                max_val = coord[i]
        ans *= (max_val - min_val)
    return ans

@njit
def find_skeleton_distance_to_pole(skeleton: np.ndarray, image_center: tuple[int]):
    min_distance = 9e10
    selected_point = skeleton[0]
    for point in skeleton:
        new_distance = (point[2]-image_center[2])**2 + (point[1]-image_center[1])**2
        if new_distance < min_distance:
            min_distance = new_distance
            selected_point = point
    return (min_distance, selected_point)

def find_closest_skeletons(image: np.ndarray, image_center: tuple[int]) -> list[tuple[np.ndarray]]:
    timer = time.time()
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = nd.label(image, structure=structure)
    print(f'\t--labeling: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    skeletons_width_distances = []
    for label_num in range(1, num_features + 1):
        flat_indices = np.flatnonzero(labeled_array == label_num)
        blob_indices = np.unravel_index(flat_indices, labeled_array.shape)
        blob_coords = np.column_stack(blob_indices)

        distance, point = find_skeleton_distance_to_pole(blob_coords, image_center)
        skeletons_width_distances.append((blob_coords, point, distance))
    print(f'\t--dividing blobs: {round(time.time() - timer, 3)}s')
    timer = time.time()

    skeletons_width_distances.sort(key = lambda x: x[2])
    return [(i[0], i[1]) for i in skeletons_width_distances[:2]]

def floodfill_nearby_skeletons(image: np.ndarray, skeletons: list):
    image = image.astype(int)
    for skeleton in skeletons:
        floodfill_3d(image, new_value=2, seed=tuple(skeleton[1]))
    return image == 2

def cont_ratio(contour: np.ndarray):
    rect = cv.boundingRect(contour)
    ratio = rect[2] / rect[3]
    return 1/ratio if ratio > 1 else ratio

def remove_large_slices_3d(image: np.ndarray, threshold: int):
    thresholded = np.where(image > threshold, 255, 0).astype(np.uint8)
    for layer_id, layer in enumerate(thresholded):
        contours, _ = cv.findContours(layer, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cont_ratio(c) > 0.6 and cv.contourArea(c) > 2000]
        cv.drawContours(layer, large_contours, -1, 0, -1)
        image[layer_id] = layer
    return image

@njit
def custom_floodfill_3d(image: np.ndarray, seed_point: tuple[int], new_value: int = 100, offsets: tuple = (-5, 5)):
    image = image.copy()
    x_size, y_size, z_size = image.shape
    x, y, z = seed_point

    start_value = int(np.average(image[x, y-25:y+25, z-25:z+25]))
    lower_bound, upper_bound = start_value + offsets[0], start_value + offsets[1]
    
    stack = [(x, y, z)]
    visited = np.zeros(image.shape, dtype=np.bool_)

    while stack:
        cx, cy, cz = stack.pop()
        if not (0 <= cx < x_size and 0 <= cy < y_size and 0 <= cz < z_size)\
            or visited[cx, cy, cz]:
            continue

        current_value = image[cx, cy, cz]

        if lower_bound <= current_value <= upper_bound:
            image[cx, cy, cz] = new_value
            visited[cx, cy, cz] = True
            if cx + 1 < x_size: stack.append((cx + 1, cy, cz))
            if cx - 1 >= 0: stack.append((cx - 1, cy, cz))
            if cy + 1 < y_size: stack.append((cx, cy + 1, cz))
            if cy - 1 >= 0: stack.append((cx, cy - 1, cz))
            if cz + 1 < z_size: stack.append((cx, cy, cz + 1))
            if cz - 1 >= 0: stack.append((cx, cy, cz - 1))

    return visited