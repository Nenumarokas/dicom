#!/usr/bin/env python

from tqdm import tqdm
from cv3d import make_skeleton
from ply_creation_lib import create_new_ply, write_to_ply, finalize_ply
from scipy import ndimage as nd
import pydicom as dicom
import numpy as np
import struct
import pickle
import time
import os

print('\n \n ')

#region DICOM
def calculate_height_to_width_ratio(folder: str, file1: str, file2: str) -> float:
    data1 = dicom.dcmread(f'{folder}\\{file1}')
    data2 = dicom.dcmread(f'{folder}\\{file2}')
    height_1 = data1.get((0x0020, 0x0032)).value[2]
    height_2 = data2.get((0x0020, 0x0032)).value[2]
    height_diff = abs(height_2 - height_1)
    pixel_spacing = data1.get((0x0028, 0x0030)) # these should be equal
    height_ratio = height_diff / pixel_spacing[0]
    return height_ratio
#endregion DICOM

def floodfill_3d(image: np.ndarray, new_value: int = 2, seed: tuple = (0, 0, 0)) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=int)
    flood_filled, _ = nd.label(image == image[seed], structure=structure)
    image[flood_filled == flood_filled[seed]] = new_value
    return image

def filter_data(mask: np.ndarray) -> np.ndarray:
    mx, my, mz = mask.shape
    
    outside_negative = floodfill_3d(mask.astype(int), -1)
    
    inside_mask = outside_negative > -1
    flooded = floodfill_3d(inside_mask.astype(int), new_value=2, seed=(mx//2, my//2, mz//2))
    
    inside_mask = flooded > 1
    return inside_mask

def prepare_data(data: list[dicom.FileDataset], threshold: tuple[int]) -> list[tuple]:
    timer = time.time()
    image = np.array([dicom.pixel_array(datum) for datum in data])
    print(f'----3D array creation: {round(time.time() - timer, 3)}s')
    timer = time.time()

    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    print(f'----normalizing: {round(time.time() - timer, 3)}s')
    timer = time.time()

    mask = image > threshold[0]
    print(f'----thresholding: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    mask = filter_data(mask)
    print(f'----filtering data: {round(time.time() - timer, 3)}s')
    timer = time.time()




    crop_mask = np.zeros_like(mask)
    crop_mask[:, 120:250, 120:250] = True
    mask = np.bitwise_and(mask, crop_mask)
    exported = np.where(mask, image, 0)
    with open('temp\\data.pkl', 'wb') as f:
        pickle.dump(exported, f)
    # exit()




    
    # mask = make_skeleton(mask)
    # print(f'----making skeleton: {round(time.time() - timer, 3)}s')
    # timer = time.time()
    
    hist, bins = np.histogram(image[mask], bins=256, range=(threshold[0], 255))
    cdf = hist.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])
    colors_normalized = np.interp(image[mask], bins[:-1], cdf_normalized)
    print(f'----standartizing colors: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    data_as_tuples = np.column_stack((*np.nonzero(mask), colors_normalized))
    print(f'----tupling: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    return data_as_tuples

def filter_points(data: list[tuple]):
    width = (120, 250)
    height = (120, 250)

    new_data = []
    for datum in data:
        if height[0] < datum[2] < height[1]:
            if width[0] < datum[1] < width[1]:
                new_data.append(datum)
    return new_data


def all_at_once(result_file: str, data: list[dicom.FileDataset], height_ratio: float, threshold: tuple[int]) -> None:
    assert threshold[0] > 0 and threshold[0] < 256
    assert threshold[1] > 0 and threshold[1] < 256
    assert threshold[0] < threshold[1]
    """
    --threshold = 150
    
    reading files: 1.104s
    preparing ply file: 0.041s
    ----3D array creation: 0.366s
    ----normalizing: 0.917s
    ----thresholding: 0.111s
    ----standartizing colors: 0.83s
    ----tupling: 0.839s
    preparing data: 3.175s
    writing to ply: 39.981s
    finalizing ply: 0.003s
    total time taken: 43.2s
    """
    
    final_timer = time.time()
    timer = time.time()
    
    create_new_ply(result_file, colored=True)
    print(f'preparing ply file: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    data_as_tuples = prepare_data(data, threshold)
    print(f'preparing data: {round(time.time() - timer, 3)}s')
    timer = time.time()

    # data_as_tuples = filter_points(data_as_tuples)
    # print(f'filtering specific data points: {round(time.time() - timer, 3)}s')
    # timer = time.time()

    write_to_ply(result_file, data_as_tuples, height_ratio)
    print(f'writing to ply: {round(time.time() - timer, 3)}s')
    timer = time.time()

    finalize_ply(result_file, len(data_as_tuples))
    print(f'finalizing ply: {round(time.time() - timer, 3)}s')
    print(f'total time taken: {round(time.time() - final_timer, 3)}s')

if __name__ == '__main__':
    """
    total time to create a ply: <1 min, depending on the threshold
    should work with most .dcm files
    """
    timer = time.time()
    folder = f'{os.getcwd()}\\20240506'
    result_file = 'result.ply'

    files: list[str] = os.listdir(folder)
    data = [dicom.dcmread(f'{folder}\\{file}') for file in files if file.endswith('.dcm')]
    print(f'reading files: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    height_ratio = calculate_height_to_width_ratio(folder, *files[:2])
    
    all_at_once(result_file, data, height_ratio, threshold=(150, 255))
