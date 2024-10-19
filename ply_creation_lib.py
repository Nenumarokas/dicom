from tqdm import tqdm
import struct
import os

def create_new_ply(filename: str, colored: bool = False) -> None:
    """
    Deletes the file it it exists and creates a new one with a ply header.
    """
    if os.path.exists(filename):
        os.remove(filename)
    
    with open(filename,'wb') as f:
        if colored:
            f.write(
                b'ply\n'
                b'format binary_little_endian 1.0\n'
                b'element vertex 000000000\n'
                b'property float x\n'
                b'property float y\n'
                b'property float z\n'
                b'property uchar red\n'
                b'property uchar green\n'
                b'property uchar blue\n'
                b'end_header\n'
            )
        else:
            f.write(
                b'ply\n'
                b'format binary_little_endian 1.0\n'
                b'element vertex 000000000\n'
                b'property float x\n'
                b'property float y\n'
                b'property float z\n'
                b'end_header\n'
            )

def write_to_ply(result_file: str, data_as_tuples: list[tuple], height_ratio: float) -> None:
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
            color = int(point[3])
            f.write(struct.pack('<fff', *location))
            f.write(struct.pack('<BBB', color, color, color))

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