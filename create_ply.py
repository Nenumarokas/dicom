#!/usr/bin/env python

from tqdm import tqdm
import pydicom as dicom
import numpy as np
import cv2 as cv
import itertools
import struct
import numba
import time
import os

print('\n \n ')

def calculate_height_to_width_ratio(folder: str, file1: str, file2: str) -> float:
    data1 = dicom.dcmread(f'{folder}\\{file1}')
    data2 = dicom.dcmread(f'{folder}\\{file2}')
    height_1 = data1.get((0x0020, 0x0032)).value[2]
    height_2 = data2.get((0x0020, 0x0032)).value[2]
    height_diff = abs(height_2 - height_1)
    pixel_spacing = data1.get((0x0028, 0x0030)) # these should be equal
    height_ratio = height_diff / pixel_spacing[0]
    return height_ratio
 
def create_new_ply(filename: str) -> None:
    if os.path.exists(filename):
        os.remove(filename)
    
    with open(filename,'wb') as f:
        f.write(
            b'ply\n'
            b'format binary_little_endian  1.0\n'
            b'element vertex 000000000\n'
            b'property float x\n'
            b'property float y\n'
            b'property float z\n'
            b'property uchar red\n'
            b'property uchar green\n'
            b'property uchar blue\n'
            b'end_header\n'
        )

def finalize_ply(output_file: str, count: int):
    with open(output_file, 'rb+') as file:
        file.readline()
        file.readline()
        third_line_start = file.tell()
        file.seek(third_line_start)
        original_third_line = file.readline()
        
        new_line = f'element vertex 000000000\n'
        if len(new_line) != len(original_third_line):
            print("ERROR: New line must have the same number of characters as the original.")
        
        file.seek(third_line_start)
        file.write(f'element vertex {count:9}\n'.encode('utf-8'))         

def prepare_data(data: list[dicom.FileDataset], threshold: int):
    timer = time.time()
    image = np.array([dicom.pixel_array(datum) for datum in data])
    print(f'3D array creation: {round(time.time() - timer, 3)}s')
    timer = time.time()

    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255
    print(f'normalizing: {round(time.time() - timer, 3)}s')
    timer = time.time()

    mask = image > threshold
    print(f'thresholding: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    data_as_tuples = np.column_stack((*np.nonzero(mask), image[mask]))
    print(f'tupling: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    return data_as_tuples

def all_at_once(data: list[dicom.FileDataset], height_ratio: float, threshold: int) -> None:
    assert threshold > 0 and threshold < 255
    """
    threshold = 150
    
    reading files: 1.282s
    3D image creation: 0.421s
    normalizing: 1.052s
    thresholding: 0.142s
    tupling: 1.156s
    writing_to_ply: 42.031s
    finalizing: 0.003s
    total time taken: 44.807s
    """
    
    final_timer = time.time()
    
    
    create_new_ply(result_file)
    data_as_tuples = prepare_data(data, threshold)
    
    with open(result_file, 'ab') as f:
        for point in tqdm(data_as_tuples, ncols=100):
            location = list(map(float, [point[1], point[2], int(point[0]*height_ratio)]))
            color = int(point[3])
            f.write(struct.pack('<fff', *location))
            f.write(struct.pack('<BBB', color, color, color))
    print(f'writing_to_ply: {round(time.time() - timer, 3)}s')
    timer = time.time()

    finalize_ply(result_file, len(data_as_tuples))
    print(f'finalizing: {round(time.time() - timer, 3)}s')
    print(f'total time taken: {round(time.time() - final_timer, 3)}s')

if __name__ == '__main__':
    """
    total time to create a ply: ~1 min, depending on the threshold
    should work with most .dcm files
    """
    timer = time.time()
    folder = f'{os.getcwd()}\\20240506'
    result_file = 'result.ply'

    files = os.listdir(folder)
    data = [dicom.dcmread(f'{folder}\\{file}') for file in files]
    print(f'reading files: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    height_ratio = calculate_height_to_width_ratio(folder, *files[:2])
    
    
    all_at_once(data, height_ratio, threshold=150)
