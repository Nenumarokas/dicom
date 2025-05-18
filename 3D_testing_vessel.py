from tqdm import tqdm
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ply_creation_lib import create_ply
import matplotlib.pyplot as plt
import monai.losses as losses
import torch.optim as optim
import numpy as np
import random
import shutil
import torch
import time
import copy
import json
import os

def read_annotation(annotation_file: str) -> np.ndarray:
    return np.load(annotation_file)

def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target):
    smooth = 1e-6
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(pred, target):
    smooth = 1e-6
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)

class DataLoaderDataset(Dataset):
    def __init__(self, disks: list[str], filenames: list):
        self.data_folders = disks
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        disk = random.choice(self.data_folders)

        filename = self.filenames[idx]
        scan = np.load(f'{disk}/scans/{filename}')
        mask = np.load(f'{disk}/masks/{filename}')
        scan = torch.from_numpy(scan).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return (scan, mask)

def prepare_data(folders: list[str], train_size: float, test_size: float, batch_size: int, workers: int):
    filenames = os.listdir(f'{folders[0]}/scans')

    random.seed(11)
    random.shuffle(filenames)

    validation_filenames, training_filenames = train_test_split(
        filenames,
        test_size=int(len(filenames)*train_size),
        random_state=11)
    
    validation_filenames, testing_filenames = train_test_split(
        validation_filenames,
        test_size=int(len(validation_filenames)*test_size),
        random_state=11)

    print(len(testing_filenames))

    train_dataset = DataLoaderDataset(folders, training_filenames)
    val_dataset = DataLoaderDataset(folders, validation_filenames)
    test_dataset = DataLoaderDataset(folders, testing_filenames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    return (train_loader, val_loader, test_loader)

def is_gpu_available() -> bool:
    return torch.cuda.is_available()

def load_model(model_location: str, model: UNet, device: str) -> UNet:
    model.load_state_dict(
            torch.load(
                f'{model_location}/model.pth',
                map_location=torch.device(device)))
    model.eval()
    return model

def prepare_vessel_model(vessel_model_location: str, device: str):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=1
    ).to(device)
    return load_model(vessel_model_location, model, device)

if __name__ == '__main__':
    train_size = 0.8
    test_size = 0.5
    new_data_size = 128
    augment_count = 4
    from_annotations = True

    device = 'cuda' if is_gpu_available() else 'cpu'
    vessel_model_location = f'{os.getcwd()}/models/train35_vessel_16-256_diceloss'
    model = prepare_vessel_model(vessel_model_location, device)
    criterion = losses.DiceLoss(sigmoid=True)
    
    selected_data_folder = 'nii_augmented'
    location_folders = [
        f'/mnt/c/Users/mariu/Downloads/train_dataset/my_dataset/{selected_data_folder}',
        f'/mnt/d/dicom/my_dataset/{selected_data_folder}']
    num_epochs = 100
    batch_size = 8
    workers = 4

    timer = time.time()
    _, _, test_loader = prepare_data(location_folders, train_size, test_size, batch_size, workers)
    print(f'preparing data: {round(time.time()-timer, 2)}s')
    timer = time.time()

    model.eval()
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    test_precision = 0.0
    test_recall = 0.0

    with tqdm(test_loader, desc='Testing', leave=False) as vbar:
        with torch.no_grad():
            for test_scans, test_masks in vbar:
                test_scans, test_masks = test_scans.to('cuda'), test_masks.to('cuda')
                preds = model(test_scans)

                test_loss += criterion(preds, test_masks).item()
                preds = (torch.sigmoid(preds) > 0.5).float()

                preds_flat = preds.view(-1)
                masks_flat = test_masks.view(-1)

                test_dice += dice_coefficient(preds_flat, masks_flat).item()
                test_iou += iou_score(preds_flat, masks_flat).item()
                test_precision += precision_score(preds_flat, masks_flat).item()
                test_recall += recall_score(preds_flat, masks_flat).item()
                
    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    test_iou /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)

    print(f'Loss:     {test_loss:.4f}')
    print(f'Dice:     {test_dice:.4f}')
    print(f'IoU:      {test_iou:.4f}')
    print(f'Precision:{test_precision:.4f}')
    print(f'Recall:   {test_recall:.4f}')
