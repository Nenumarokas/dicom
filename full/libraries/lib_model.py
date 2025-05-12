from tqdm import tqdm
from skimage.transform import resize
from monai.networks.nets import UNet
from numpy.lib.stride_tricks import sliding_window_view
import pydicom as dicom
import numpy as np
import torch
import copy
import os

def is_gpu_available() -> bool:
    return torch.cuda.is_available()

def load_model(model_location: str, model: UNet, device: str) -> UNet:
    model.load_state_dict(
            torch.load(
                f'{model_location}/model.pth',
                map_location=torch.device(device)))
    model.eval()
    return model

def prepare_datum(datum: np.ndarray) -> torch.Tensor:
    datum = downscale(datum)
    device = 'cuda' if is_gpu_available() else 'cpu'
    input_tensor = torch.tensor(datum, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    return input_tensor

def downscale(data: np.ndarray):
    return resize(
        data,
        output_shape=(128, 128, 128),
        order=1,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.float64)

def upscale(data: np.ndarray):
    return (resize(
        data.astype(np.float32),
        output_shape=(512, 512, 512),
        order=1,
        preserve_range=True,
        anti_aliasing=False
    ) > 0.5).astype(np.float64)

def single_datum_inference(model: UNet, input_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(input_tensor)
    prob_map = torch.sigmoid(output)
    binary_mask = (prob_map > 0.5).float()
    array = binary_mask.detach().cpu().numpy()
    return upscale(array[0, 0])

def batch_inference(model: UNet, data: tuple[list, list]):
    device = 'cuda' if is_gpu_available() else 'cpu'
    calcinate_scattered = []
    inference_indeces = []
    for i, mask in enumerate(tqdm(data[1], desc='Detecting calcinates... ', ncols=100, leave=False)):
        calcinate_scattered.append(np.zeros_like(mask))
        if len(np.unique(mask)) == 2:
            inference_indeces.append(i)

    batch = np.stack([data[0][i][np.newaxis, ...] for i in inference_indeces])
    batch_tensor = torch.from_numpy(batch).float().to(device)

    with torch.no_grad():
        output_tensor = model(batch_tensor)

    for i, patch in enumerate(output_tensor):
        iid = inference_indeces[i]
        calcinate_scattered[iid] = (torch.sigmoid(patch) > 0.5).float().cpu().numpy()[0]
    return calcinate_scattered

def extract_patches(image: np.ndarray, vessel_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colored_vessels = np.where(vessel_mask, image, 0).astype(np.float32)
    colored_vessels = np.clip(colored_vessels, 0, 1000)

    colored_patches = sliding_window_view(colored_vessels, (64,)*3)[::64, ::64, ::64]
    mask_patches = sliding_window_view(vessel_mask, (64,)*3)[::64, ::64, ::64]
    colored_patches = colored_patches.reshape(-1, 64, 64, 64)
    mask_patches = mask_patches.reshape(-1, 64, 64, 64)

    return (colored_patches, mask_patches)

def combine_patches(patches: list[np.ndarray]):
    blocks = np.array(patches)
    grid = blocks.reshape(8, 8, 8, 64, 64, 64)

    depth_slices = []
    for z in range(8):
        rows = []
        for y in range(8):
            row = [grid[z, y, x] for x in range(8)]
            rows.append(np.concatenate(row, axis=2))
        depth_slices.append(np.concatenate(rows, axis=1))
    full_volume = np.concatenate(depth_slices, axis=0)
    return full_volume

def resize_input(input_datum: np.ndarray) -> np.ndarray:
    if all(i == 512 for i in input_datum.shape):
        return input_datum

    z_val, y_val, x_val = input_datum.shape
    if y_val == 512 and x_val == 512:
        new_array = np.zeros((512, 512, 512))
        if z_val < 512:
            new_array[:z_val,:,:] = input_datum
            input_datum = new_array
        elif z_val > 512:
            new_array = input_datum[:512, :, :]
            input_datum = new_array
    else:
        input_datum = resize(
            input_datum,
            output_shape=(512, 512, 512),
            order=1,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.float32)
    return input_datum

def read_dicom(input_folder: str) -> np.ndarray:
    files: list[str] = os.listdir(input_folder)
    data = [dicom.dcmread(f'{input_folder}/{file}') for file in files if file.endswith('.dcm')]
    image = np.array([dicom.pixel_array(datum) for datum in data])
    return image

def rescale_image_colors(image: np.ndarray, min_scale: int = 100, max_scale: int = 1000) -> np.ndarray:
    scaled_image = image - min_scale
    scaled_image = np.clip(scaled_image, 0, max_scale - min_scale)
    return scaled_image / (max_scale - min_scale)

def recolored_image(image: np.ndarray) -> np.ndarray:
    image = copy.deepcopy(image)
    min_val, max_val = np.min(image), np.max(image) 
    image = (image - min_val) / (max_val - min_val) * 255
    return image