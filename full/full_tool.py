import sys
import os
sys.path.append(f'{os.getcwd()}/libraries')

from monai.networks.nets import UNet
from libraries.lib_mpr import create_mpr
from libraries.lib_skeleton3d import get_branches
from libraries.lib_model import *
import numpy as np
import time

def prepare_vessel_model(vessel_model_location: str) -> UNet:
    device = 'cuda' if is_gpu_available() else 'cpu'
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=1
    ).to(device)
    return load_model(vessel_model_location, model, device)

def prepare_calcinate_model(calcinate_model_location: str) -> UNet:
    device = 'cuda' if is_gpu_available() else 'cpu'
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=1
    ).to(device)
    return load_model(calcinate_model_location, model, device)

def get_vessel_annotation(data_name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(f'input_data/annotations/{data_name}_annotation.npy')

if __name__ == '__main__':
    # ['20241209_17', '20250222_37', '20250224_45', '20250224_46']
    input_folder = '20250224_45'
    mpr_output_folder = 'other_mpr_results'
    mpr_rotations = 24
    show_mode = True

    vessel_model_location = f'models/train36_vessel'
    calcinate_model_location = f'models/train32_calcinate_16_32_64chan_diceloss'
    timer = time.time()


    # Read and prepare images
    print('\nPreparing images...   ', end='')
    base_image = read_dicom(f'input_data/{input_folder}')
    image = resize_input(base_image)
    colored_image = recolored_image(image)
    normalized_image = rescale_image_colors(image)
    print(f'done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    # Load detection models
    print('Loading models...   ', end='', flush=True)
    vessel_model = prepare_vessel_model(vessel_model_location)
    calcinate_model = prepare_calcinate_model(calcinate_model_location)
    print(f'done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    # Preparing scan for detection
    print('Preparing datum...   ', end='', flush=True)
    datum = downscale(normalized_image)
    datum = prepare_datum(datum)
    print(f'done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    if not show_mode:
        # Detect vessels using a detection model (unused in show mode)
        print('Detecting vessels...   ', end='', flush=True)
        vessel_result = single_datum_inference(vessel_model, datum).astype(bool)
        vessel_result = upscale(vessel_result)
        print(f'done. ({round(time.time()-timer, 2)}s)')
        timer = time.time()


    # Read vessel annotation (used only in show mode)
    print('Reading vessel annotation...   ', end='', flush=True)
    vessel_annotation = get_vessel_annotation(input_folder)
    vessel_annotation = resize_input(vessel_annotation)
    print(f'done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    # Divide the vessel volume into smaller pieces for the calcinate model
    print('Dividing vessels...   ', end='', flush=True)
    if show_mode:
        # Uses pre-annotated vessels
        patches = extract_patches(image, vessel_annotation)
        branches = get_branches(vessel_annotation)
    else:
        # Uses model-detected vessels
        patches = extract_patches(image, vessel_result)
        branches = get_branches(vessel_result)
    print(f'done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    # Detect calcinates using a detection model
    print('Detecting calcinates...   ', end='', flush=True)
    patches = batch_inference(calcinate_model, patches)
    calcinate_result = combine_patches(patches).astype(bool)
    print(f'Detecting calcinates...   done. ({round(time.time()-timer, 2)}s)')
    timer = time.time()


    # Save MPR images to a specified folder
    print('Saving MPR...   ', end='', flush=True)
    if show_mode:
        # Save image without marking the calcinates
        create_mpr(colored_image, calcinate_result, branches, f'results/{mpr_output_folder}_base', mpr_rotations, colored=False)
    create_mpr(colored_image, calcinate_result, branches, f'results/{mpr_output_folder}_colored', mpr_rotations, colored=True)
    print(f'done. ({round(time.time()-timer, 2)}s)')
