from tqdm import tqdm
from array_lib import *
from ply_creation_lib import create_ply
import pydicom as dicom
import numpy as np
import cv2 as cv
import pickle
import copy
import os

def read_annotations(annotation_file: str):
    return np.load(annotation_file)

if __name__ == '__main__':
    annotation_file = f'{os.getcwd()}/20250222_37_annotation.npy'
    
    annotation = read_annotations(annotation_file)
    
    create_ply(annotation, 'annotations.ply')