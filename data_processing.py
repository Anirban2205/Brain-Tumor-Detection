import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import random
import cv2
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as tt
import albumentations as A
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class BrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0])
        image = np.array(image)/255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask)/255.

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.float32)
        image = tt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2,0,1))
        mask = torch.from_numpy(mask).type(torch.float32)

        return image, mask


train_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
])

val_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
    A.HorizontalFlip(p=0.5),
])

test_transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),
])

def create_custom_dataset(train_df, val_df, test_df):
    set_seed()

    train_ds = BrainDataset(train_df, train_transform)
    val_ds = BrainDataset(val_df, val_transform)
    test_ds = BrainDataset(test_df, test_transform)

    return train_ds, val_ds, test_ds


def create_dataloader(batch_size, train_ds, val_ds, test_ds):
    set_seed()
    train_dl = DataLoader(train_ds,
                        batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True)
    set_seed()
    val_dl = DataLoader(val_ds,
                        batch_size,
                        num_workers=2,
                        pin_memory=True)
    test_dl = DataLoader(val_ds,
                        batch_size,
                        num_workers=2,
                        pin_memory=True)
    
    return train_dl, val_dl, test_dl