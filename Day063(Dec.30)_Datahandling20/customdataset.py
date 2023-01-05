import glob
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
# from torchvision.transforms import  transforms



class customDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_paths = glob.glob(os.path.join(img_path,"*","*.jpg"))
        # print(self.file_path)
        self.class_names = os.listdir(img_path)
        self.class_names.sort()
        self.transforms = transform
        self.img_paths.sort()
        self.labels= []

        for path in self.img_paths:
            self.labels.append(self.class_names.index(path.split('\\')[1]))
        # print(self.labels)
        self.labels = np.array(self.labels)


    def __getitem__(self, item):
        image_path = self.img_paths[item]
        image = cv2.imread(image_path)
        # print(image)
        label = self.labels[item]
        label = torch.tensor(label)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        print(image_path, label)


    def __len__(self):
        return len(self.img_paths)

test = customDataset("./dataset/train")
for i in test:
    pass

