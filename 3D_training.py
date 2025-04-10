from tqdm import tqdm
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import monai.losses as losses
import torch.optim as optim
import numpy as np
import torch
import time
import copy
import json
import os

def get_prepared_data(folder: str) -> list[dict[str, np.ndarray]]:
    filenames = os.listdir(f'd:\\dicom\\my_dataset\\{folder}\\scans')
    filenames.sort(key = lambda x: int(x.split('.')[0]))

    data = []
    for filename in tqdm(filenames, desc='loading data'):
        scan = np.load(f'd:\\dicom\\my_dataset\\{folder}\\scans\\{filename}').astype(np.int16)
        mask = np.load(f'd:\\dicom\\my_dataset\\{folder}\\masks\\{filename}').astype(np.int16)
        data.append((scan, mask))
    return data

def read_annotation(annotation_file: str) -> np.ndarray:
    return np.load(annotation_file)

def dice_coefficient(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class DataLoaderDataset(Dataset):
    def __init__(self, data_folder: str, filenames: list):
        self.data_folder = data_folder
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        scan = np.load(f'{self.data_folder}/scans/{filename}')
        mask = np.load(f'{self.data_folder}/masks/{filename}')
        scan = torch.tensor(scan, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return (scan, mask)

def prepare_data(folder: str, train_size: float, test_size: float):
    filenames = os.listdir(f'{folder}/scans')

    validation_filenames, training_filenames = train_test_split(
        filenames,
        test_size=int(len(filenames)*train_size),
        random_state=11)
    
    validation_filenames, testing_filenames = train_test_split(
        validation_filenames,
        test_size=int(len(validation_filenames)*test_size),
        random_state=11)


    train_dataset = DataLoaderDataset(folder, training_filenames)
    val_dataset = DataLoaderDataset(folder, validation_filenames)
    test_dataset = DataLoaderDataset(folder, testing_filenames)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    return (train_loader, val_loader, test_loader)

if __name__ == '__main__':
    train_size = 0.8
    test_size = 0.5
    new_data_size = 128
    augment_count = 4
    from_annotations = True

    model_output_folder = f'{os.getcwd()}/models'    
    selected_data_folder = 'nii_augmented'
    location_folder = f'/mnt/d/dicom/my_dataset/{selected_data_folder}'
    num_epochs = 1

    print(os.getcwd())


    timer = time.time()
    train_loader, val_loader, test_loader = prepare_data(location_folder, train_size, test_size)
    print(f'preparing data: {round(time.time()-timer, 2)}')
    timer = time.time()


    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=1
    ).to('cuda')

    criterion = losses.DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f'loading model: {round(time.time()-timer, 2)}')
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

            outputs_bin = torch.sigmoid(outputs) > 0.5
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
                    val_dice += dice_coefficient(test_outputs, test_masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        val_results = f'validation - loss: {val_loss:.4f}, dice: {val_dice:.4f}'

        metrics['val_loss'].append(val_loss)
        metrics['val_dice'].append(val_dice)

        print(f'{train_results} | {val_results}', end='')


    print(f'total time taken to train: {round(time.time()-timer, 2)}')



    last_model_number = max(int(i[5:]) for i in os.listdir(model_output_folder))
    selected_folder = f'{model_output_folder}\\train{last_model_number+1}'
    os.mkdir(selected_folder)

    torch.save(model.state_dict(), f'{selected_folder}\\model.pth')
    with open(f'{selected_folder}\\metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'saved as \"{selected_folder}\"')
