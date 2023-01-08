import os
import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

train_path = "./dataset/train"
val_path = "./dataset/val"
test_path = "./dataset/test"

class CustomDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.all_img_path = glob.glob(os.path.join(img_path,"*","*.png"))
        # print(self.img_path)   # >>> './dataset/train\\water\\SeaLake_997.png'
        self.transform = transform
        # self.img_list = []
        # for img_path in self.all_img_path:
        #     self.img_list.append(Image.open(img_path))
        # 라벨 리스트 저장 !!
        self.label_dict = {"cloudy":0,"desert":1,"green_area":2,"water":3}

    def __getitem__(self, item):
        # img = self.img_list[item]
        # # print(img)
        img_path = self.all_img_path[item]
        # print(img_path)       # ./dataset/train\cloudy\train_10021.png
        label_temp = img_path.split("\\")
        label = self.label_dict[label_temp[1]]
        # print(label_temp)       # ['./dataset/train', 'cloudy', 'train_10021.png']
        # print(label_temp[1])    # cloudy
        # print(label)            # 0
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
            # 여기서 transform 된 이미지를 저장할 수 있음
        return img, label

    def __len__(self):
        return len(self.all_img_path)

# if __name__ == '__main__':
#     test = CustomDataset(train_path, transform=None)
#     for i in test:
#         print(i)