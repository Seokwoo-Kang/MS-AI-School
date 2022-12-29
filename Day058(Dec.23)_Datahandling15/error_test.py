from dataset_custom import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
import hy_params
from torchvision import models
import torch.nn as nn
from utils import train, validate, save_model
import os

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train Aug
train_transform = A.Compose([
    A.Resize(height=224,width=224),
    ToTensorV2()
])

# Val Aug
val_transform = A.Compose([
    A.Resize(height=224,width=224),
    ToTensorV2()
])

# dataset
train_dataset = custom_dataset("./data/train", transform=train_transform)
val_dataset = custom_dataset("./data/val", transform=val_transform)

# print(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=hy_params.batch_size, shuffle=True)
print(train_loader)