#!/usr/bin/env python

from tqdm import tqdm
from scipy import ndimage as nd
from ply_creation_lib import create_new_ply, write_to_ply, finalize_ply
import pydicom as dicom
import numpy as np
import struct
import pickle
import time
import os

class BoxArray:
    def __init__(self, data: np.ndarray, box_dims: tuple[int]):
        self.box_count = 0
        self.boxes = []

    def divide(self, data: np.ndarray, box_dims: tuple[int]):
        """
        divide data into boxes of dims provided.
        If the data does not match the multiple of boxes, pad with zeroes.
        """
        pass

class Box:
    def __init__(self, location: int, data: list):
        self.location = location
        self.data = data
        
        self.neighbours = []

    def set_neighbours(self, neighbours: list['Box']):
        self.neighbours = neighbours

def get_data(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def divide_into_boxes(data: np.ndarray):
    print(len(data))
    print(len(data[0]))
    print(len(data[0,0]))

def create_ply(data: np.ndarray, output_file: str):
    mask = data > 0
    data_as_tuples = np.column_stack((*np.nonzero(mask), data[mask]))
    print(len(data_as_tuples))
    create_new_ply(output_file, colored=False)
    write_to_ply(output_file, data_as_tuples, height_ratio=0.66)
    finalize_ply(output_file, len(data_as_tuples))

if __name__ == '__main__':
    input_file = 'temp\\data.pkl'
    small_ply = 'small.ply'
    
    timer = time.time()
    data = get_data(input_file)
    print(f'data_read: {round(time.time() - timer, 3)}s')
    timer = time.time()

    create_ply()
print(np.count_nonzero(data))