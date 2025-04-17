from tqdm import tqdm
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import monai.losses as losses
import torch.optim as optim
import numpy as np
import random
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


    train_dataset = DataLoaderDataset(folders, training_filenames)
    val_dataset = DataLoaderDataset(folders, validation_filenames)
    test_dataset = DataLoaderDataset(folders, testing_filenames)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    return (train_loader, val_loader, test_loader)

if __name__ == '__main__':
    train_size = 0.8
    test_size = 0.5
    new_data_size = 128
    augment_count = 4
    from_annotations = True

    model_output_folder = f'{os.getcwd()}/models'    
    selected_data_folder = 'nii_augmented'
    location_folders = [
        f'/mnt/c/Users/mariu/Downloads/train_dataset/my_dataset/{selected_data_folder}',
        f'/mnt/d/dicom/my_dataset/{selected_data_folder}']
    num_epochs = 100
    batch_size = 8
    workers = 4

    print(os.getcwd())


    timer = time.time()
    train_loader, val_loader, test_loader = prepare_data(location_folders, train_size, test_size, batch_size, workers)
    print(f'preparing data: {round(time.time()-timer, 2)}s')
    timer = time.time()



    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=1
    ).to('cuda')

    # dice_loss = losses.DiceLoss(sigmoid=True)
    # bce_loss = torch.nn.BCEWithLogitsLoss()
    # criterion = lambda pred, target: 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
    criterion = losses.DiceLoss(sigmoid=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f'loading model: {round(time.time()-timer, 2)}s')
    timer = time.time()

    metrics = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        for scans, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):

            scans, masks = scans.to('cuda'), masks.to('cuda')

            optimizer.zero_grad()
            outputs = model(scans)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            outputs_bin = (torch.sigmoid(outputs) > 0.5).float()
            dice = dice_coefficient(outputs_bin, masks)

            epoch_loss += loss.item()
            epoch_dice += dice.item()

        epoch_loss /= len(train_loader)
        epoch_dice /= len(train_loader)

        train_results = f'training - loss: {epoch_loss:.4f}, dice: {epoch_dice:.4f}'

        metrics['train_loss'].append(epoch_loss)
        metrics['train_dice'].append(epoch_dice)






        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with tqdm(val_loader, desc='Validating', leave=False) as vbar:
            with torch.no_grad():
                for test_scans, test_masks in vbar:
                    test_scans, test_masks = test_scans.to('cuda'), test_masks.to('cuda')
                    test_outputs = model(test_scans)

                    val_loss += criterion(test_outputs, test_masks).item()
                    test_outputs = torch.sigmoid(test_outputs) > 0.5
                    val_dice += dice_coefficient(test_outputs, test_masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # scheduler.step(val_loss)

        val_results = f'validation - loss: {val_loss:.4f}, dice: {val_dice:.4f}'

        metrics['val_loss'].append(val_loss)
        metrics['val_dice'].append(val_dice)

        print(f'{train_results} | {val_results}', end='')


    time_taken = time.time() - timer
    print(f'\ntotal time taken to train: {round(time_taken)}seconds')
    print(f'\ntotal time taken to train: {round(time_taken/3600, 2)}hours')



    last_model_number = max(int(i.split('_')[0][5:]) for i in os.listdir(model_output_folder))
    selected_folder = f'{model_output_folder}/train{last_model_number+1}'
    os.mkdir(selected_folder)

    torch.save(model.state_dict(), f'{selected_folder}/model.pth')
    with open(f'{selected_folder}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'saved as \"{selected_folder}\"')

    with open(f'{selected_folder}/time.txt', 'w') as f:
        f.write(f'{time_taken}')

