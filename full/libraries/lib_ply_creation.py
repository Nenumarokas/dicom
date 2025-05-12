from tqdm import tqdm
from lib_skeleton3d import Skeleton
import pydicom as dicom
import numpy as np
import struct
import os

def create_new_ply(filename: str, normaled: bool, colored: bool) -> None:
    """
    Deletes the file it it exists and creates a new one with a ply header.
    """
    if os.path.exists(filename):
        os.remove(filename)
    
    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        'element vertex 000000000\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    normals = (
        'property float nx\n'
        'property float ny\n'
        'property float nz\n'
    )
    colors = (
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
    )
    header_end = 'end_header\n'
    
    if normaled: header += normals
    if colored: header += colors
    header += header_end
    
    with open(filename,'wb') as f:
        f.write(header.encode())

def write_to_ply(result_file: str, data_as_tuples: list[tuple], height_ratio: float, color: int = -1) -> None:
    """
    Data format must be the tuple of:
        - depth value
        - height value
        - width value
        - color value
    """
    with open(result_file, 'ab') as f:
        for point in tqdm(data_as_tuples, ncols=100):
            location = list(map(float, [point[2], point[1], -point[0]*height_ratio]))
            if color < 0:
                selected_color = int(point[3])
            else:
                selected_color = color
            if selected_color < 0 or selected_color > 255:
                selected_color = 0
            
            f.write(struct.pack('<fff', *location))
            f.write(struct.pack('<BBB', selected_color, selected_color, selected_color))
            
def write_to_ply_normals(result_file: str, data: Skeleton, height_ratio: float) -> None:
    with open(result_file, 'ab') as f:
        for point in tqdm(data, ncols=100):
            coords = point.coordinates
            normal = point.normal
            point_location = list(map(float, [coords[2], coords[1], -coords[0]*height_ratio]))
            normal_location = list(map(float, [normal[2], normal[1], -normal[0]*height_ratio]))
            f.write(struct.pack('<fff', *point_location))
            f.write(struct.pack('<fff', *normal_location))

def finalize_ply(output_file: str, count: int) -> None:
    """
    Updates the .ply header vertex count
    """
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

def create_ply_normals(data: Skeleton, output_file: str) -> None:
    """
    Creates a ply from a list of coordinate and normal tuples and puts it into a specified file.
    Height distances are incorrect without specifying the height between pixels.
    """
    if not output_file:
        raise ValueError('\"output_file\" parameter cannot be an empty string')
    
    if len(data) == 0:
        raise ValueError("\"data\" parameter cannot be empty")
    
    if not output_file.endswith('.ply'):
        output_file += '.ply'
        
    create_new_ply(output_file, normaled=True, colored=False)
    write_to_ply_normals(output_file, data, height_ratio=0.66)
    finalize_ply(output_file, len(data))
    
def create_ply(data: np.ndarray, output_file: str, preserve_color: bool = False) -> None:
    """
    Creates a ply from a 3D numpy array and puts it into a specified file.
    Height distances are incorrect without specifying the height between pixels.
    """
    if not output_file:
        raise ValueError('\"output_file\" parameter cannot be an empty string')
    
    if not np.any(data):
        raise ValueError("\"data\" parameter cannot be empty")
    
    if not output_file.endswith('.ply'):
        output_file += '.ply'
    
    mask = data > 0
    
    data_as_tuples = np.column_stack((*np.nonzero(mask), data[mask]))
    create_new_ply(output_file, normaled=False, colored=True)
    write_to_ply(output_file, data_as_tuples, height_ratio=0.66, color=-1 if preserve_color else 255)
    finalize_ply(output_file, len(data_as_tuples))
    
def calculate_height_to_width_ratio(folder: str, file1: str, file2: str) -> float:
    # height_ratio = calculate_height_to_width_ratio(input_folder, *files[:2])
    data1 = dicom.dcmread(f'{folder}\\{file1}')
    data2 = dicom.dcmread(f'{folder}\\{file2}')
    height_1 = data1.get((0x0020, 0x0032)).value[2]
    height_2 = data2.get((0x0020, 0x0032)).value[2]
    height_diff = abs(height_2 - height_1)
    pixel_spacing = data1.get((0x0028, 0x0030)) # x and y distances between pixels should be equal
    height_ratio = height_diff / pixel_spacing[0]
    return height_ratio
    
def create_ply_tupled(data_as_tuples: list[tuple], output_file: str, height_ratio: float):
    create_new_ply(output_file, normaled=False, colored=True)
    write_to_ply(output_file, data_as_tuples, height_ratio=height_ratio)
    finalize_ply(output_file, len(data_as_tuples))
