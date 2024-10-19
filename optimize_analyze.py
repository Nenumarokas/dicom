#!/usr/bin/env python

from tqdm import tqdm
from scipy import ndimage as nd
from ply_creation_lib import create_new_ply, write_to_ply, finalize_ply
import numpy as np
import pickle
import time
import os

print('\n \n ')

#region classes
class BoxArray:
    def __init__(self, data: np.ndarray, box_dims: int):
        count, boxes = self.divide(data, box_dims)
        
        self.box_count = count
        self.boxes = boxes

    def divide(self, data: np.ndarray, box_dims: int) -> tuple[int, list['Box']]:
        """
        divide data into boxes of dims provided.
        If the data does not match the multiple of boxes, pad with zeroes.
        """
        data_shape = np.shape(data)
        box_array_shape = [int(np.ceil(data_shape[i]/box_dims)) for i in range(3)]
        
        for x in range(box_array_shape[0]-1):
            for y in range(box_array_shape[1]-1):
                for z in range(box_array_shape[2]-1):
                    print(f'({x*box_dims}-{(x+1)*box_dims}; {y*box_dims}-{(y+1)*box_dims}; {z*box_dims}-{(z+1)*box_dims})')
                    self.create_slicer((x, y, z), box_dims)
                    
                    exit()
        
        exit()
        
        boxes = [Box() for _ in range(np.prod(box_array_shape))]
        unzipped = list(zip(*np.nonzero(data)))
        print(f'total: {len(boxes)}')
        
        # need to use slicing to get the main box
        # also the surrounding boxes
        # save the current XYZ as box center part for data retrieval

        return (0, [])
    
    def create_slicer(self, coords: tuple[int], box_dims: int):
        neghbour_slicer = [slice(max((coords[i]-1)*box_dims, 0), (coords[i]+2)*box_dims) for i in range(3)]
        return neghbour_slicer
    
    def determine_box_id(self, point: tuple[int], box_array_shape: int) -> int:
        box_ids = [point[i] // box_array_shape[i] for i in range(3)]
        return box_ids[0] * box_array_shape[1] * box_array_shape[2] +\
                box_ids[1] * box_array_shape[2] +\
                box_ids[2]

class Box:
    def __init__(self):
        self.points: np.ndarray = []
        self.box_dims: tuple[int] = ()
        self.box_center: tuple[int] = ()

    def set_points(self, box: np.ndarray):
        self.box = box
        
    def get_count(self):
        return len(self.points)

#endregion classes

def get_data(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_ply(data: np.ndarray, output_file: str):
    mask = data > 0
    data_as_tuples = np.column_stack((*np.nonzero(mask), data[mask]))
    print(data_as_tuples[:4])
    create_new_ply(output_file, colored=True)
    write_to_ply(output_file, data_as_tuples, height_ratio=0.66, color=255)
    finalize_ply(output_file, len(data_as_tuples))

if __name__ == '__main__':
    input_file = 'temp\\data.pkl'
    small_ply = 'small.ply'
    
    timer = time.time()
    data = get_data(input_file)
    print(f'reading data: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    
    boxes = BoxArray(data, box_dims=100)
    print(f'dividing into boxes: {round(time.time() - timer, 3)}s')
    timer = time.time()
    
    
    exit()
    

    create_ply(data, small_ply)