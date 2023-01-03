import random

import torch
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt



class custom_dataset(Dataset):
    def __init__(self, file_path):
        # file_path -> data/train    /0/images
        self.file_path = glob.glob(os.path.join(file_path,"*","*.png"))
        # print(self.file_path)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, item):
        image_path = self.file_path[item]
        # print(image_path)
        label = image_path.split("\\")
        label = int(label[3])
        # print(label)

        mo = image_path.split("\\")
        mo = mo[2]
        img = Image.open(image_path).convert('RGB')
        # print(img)

        if mo=="train":
            if random.uniform(0,1) <0.2 or img.getbands()[0] == 'L':
                # Random gray scale from 20%
                img = img.convert('L').convert("RGB")

            if random.uniform(0,1) <0.2 :
                # Random gray scale from 20%
                gaussianBlur = ImageFilter.GaussianBlur(random.uniform(0.5, 1.2))
                img = img.filter(gaussianBlur)

        else:
            if img.getbands()[0] =='L':
                img = img.convert('L').convert('RGB')

        img = self.transform(img)
        # print(img.size(), image_path)

        return img, label

    def __len__(self):
        return  len(self.file_path)

# train_dataset = custom_dataset("D:\\data\\train\\")
#
# # print(train_dataset.__len__())
#
# _, ax = plt.subplots(2,4,figsize=(16,10))
#
# for i in range(8):
#     data = train_dataset.__getitem__(np.random.choice(range(train_dataset.__len__())))
#
#     image = data[0].cpu().detach().numpy().transpose(1,2,0)*255
#     img = image.astype(np.uint32)
#
#     label = data[1]
#
#     ax[i//4][i-(i//4)*4].imshow(image.astype("uint8"))
#     ax[i // 4][i - (i // 4) * 4].set_title(label)
# plt.show()

# test = custom_dataset("D:\Project\data\data")
# for i in test:
#     pass