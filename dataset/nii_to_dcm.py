import nibabel as nib
import numpy as np
import os

def nii_to_numpy(nii_path):
    nii_img = nib.load(nii_path)  # Load NIfTI file
    numpy_array = nii_img.get_fdata()  # Get image data as NumPy array
    return numpy_array

# Example usage
nii_file = f'{os.getcwd()}\\dataset\\1-200\\1.img.nii.gz'
array_3d = nii_to_numpy(nii_file)
print(array_3d.shape)