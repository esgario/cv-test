import os
import torch
import torch.nn as nn
import numpy as np
import imageio as io
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, root_dir='', augmentations=None, test=False):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.files = os.listdir(os.path.join(root_dir, 'noisy_data'))
        self.augmentations = augmentations
        self.test = test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        noisy_img_path = os.path.join(self.root_dir, 'noisy_data', self.files[idx])
        noisy_image = io.imread(noisy_img_path)
        
        if not self.test:
            clean_img_path = os.path.join(self.root_dir, 'clean_data', self.files[idx])
            clean_image = io.imread(clean_img_path)
        else:
            clean_image = np.ones(noisy_image.shape)
        
        if self.augmentations is not None:
            noisy_image, clean_image = self.augmentations(noisy_image, clean_image)
        
        noisy_image, clean_image = transform(noisy_image, clean_image)

        return noisy_image, clean_image


def transform(noisy_image, clean_image=None):
    if len(noisy_image.shape) > 2:
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
    noisy_image = noisy_image.astype(float) / 255.0
    noisy_image = np.expand_dims(noisy_image, 0)
    noisy_image = torch.from_numpy(noisy_image).float()

    if clean_image is not None:
        if len(clean_image.shape) > 2:
            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
        clean_image = clean_image.astype(float) / 255.0
        clean_image = np.expand_dims(clean_image, 0)
        clean_image = torch.from_numpy(clean_image).float()

        return noisy_image, clean_image
    else:
        return noisy_image


def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

            
# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    
    from augmentations import Compose, RandomHorizontallyFlip, RandomVerticallyFlip, Scale
    from augmentations import AdjustContrast, AdjustBrightness, AdjustSaturation
    import matplotlib.pyplot as plt

    bs = 4
    augmentations = Compose([Scale(512),
                             RandomHorizontallyFlip(0.5),
                             RandomVerticallyFlip(0.5),
                             AdjustContrast(0.25),
                             AdjustBrightness(0.25),
                             AdjustSaturation(0.25)])
            
    dst = CustomDataset(root_dir='../dataset/train', augmentations=augmentations)
    trainloader = DataLoader(dst, batch_size=bs, collate_fn=custom_collate, pin_memory=True)
    criterion = nn.MSELoss()

    for i, data_samples in enumerate(trainloader):
        noisy_data, clean_data = data_samples
        
        loss = criterion(noisy_data[0], clean_data[0])
        loss2 = criterion(clean_data[0], clean_data[0])
        print(loss, loss2)

        plt.imshow(noisy_data[0][0], cmap='gray', vmin=0, vmax=1)
        plt.show()
        plt.imshow(clean_data[0][0], cmap='gray', vmin=0, vmax=1)
        plt.show()
        print(noisy_data[0].shape, clean_data[0].shape)