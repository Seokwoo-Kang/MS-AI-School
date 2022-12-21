from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class MyCustomdataset(Dataset):

    def __init__(self, file_path, transforms=None):
        self.file_paths = file_path
        self.transform = transforms
        

    def __getitem__(self, index):
        image_path = self.file_paths[index]

        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        return image

    def __len(self):
        return len(self.file_paths)

torchvision_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.AutoAugment(),
    transforms.ToTensor()
])

train_dataset = MyCustomdataset(
    file_path=["./Agumentation/01.jpg"], transforms=torchvision_transform)

for i in range(100):
    sample = train_dataset[0]

plt.figure(figsize=(5,5))
plt.imshow(transforms.ToPILImage()(sample))
plt.show()
