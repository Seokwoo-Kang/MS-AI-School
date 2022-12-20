from PIL import Image
import cv2

import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from matplotlib import pyplot as plt
import albumentations
# pip install -U albumentations
from albumentations.pytorch import ToTensorV2

# 2nd. albumentations Data pipeline
# cv2로 읽어야함
class alb_cat_dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # read an image with Opencv
        image = cv2.imread(file_path)

        # BGR -> RGB로 convert
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_t = time.time()
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        total_time = (time.time()- start_t)

        return image, total_time

    def __len__(self):
        return len(self.file_paths)



# 1st. 기존 torchvision Data pipeline
# 1. dataset class -> image loader -> transform
class CatDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        # print(file_paths)
        # ./cat/cat.jpeg
        self.file_paths = file_paths
        self.transform = transform
                
    def __getitem__(self, index):
        file_path = self.file_paths[index]

        # 원래(분류)라면 image, label을 리턴했어야함 
        # Read an image with PIL
        image = Image.open(file_path)

        # transform time check
        start_time = time.time()
        if self.transform:
            image = self.transform(image)
        end_time = (time.time()- start_time)
        
        return image, end_time

    def __len__(self):
        return len(self.file_paths)


### data aug transforms
### train용 val용 따로 만들어서 관리 test는 val을 사용하면 된다 
# train 에는 random성을 넣지만 val에서는 random성을 넣지 않음
# train_transform = transforms.Compose([A+random])
# val_transform = transforms.Compose([A])

torchvision_transform = transforms.Compose([     # Compose로 한꺼번에 묶어서
    # transforms.Pad(padding=50),
    # transforms.Resize((256,256)),
    # transforms.CenterCrop(size=(30)),
    # transforms.FiveCrop(size=100),
    # transforms.Grayscale(),
    # transforms.RandomCrop(224),
    # transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.3),
    # transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
    # transforms.RandomPerspective(distortion_scale=0.7, p=0.6),
    # transforms.RandomRotation(degrees=(0,100)),
    # transforms.RandomAffine(degrees=(30,60), translate=(0.1, 0.3), scale=(0.3, 0.6)),
    # transforms.ElasticTransform(alpha=255.0), # 버전이슈
    # transforms.RandomEqualize(p=0.5),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.AutoAugment(),

    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()            
])

albumentations_transform = albumentations.Compose([
    albumentations.Resize(256,256),
    albumentations.RandomCrop(224,224),
    albumentations.HorizontalFlip(),
    albumentations.VerticalFlip(),
    ToTensorV2(),
    # albumentations.pytorch.transforms.ToTensor() 과거 버전에서 사용법
])

albumentations_transform_oneof = albumentations.Compose([
    albumentations.Resize(256,256),
    albumentations.RandomCrop(224,224),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomRotate90(p=1),
        albumentations.VerticalFlip(p=1)
    ], p=1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=1),
    ToTensorV2()
])

alb_dataset = alb_cat_dataset(file_paths=['./torch_transforms/cat.jpeg'], transform=albumentations_transform)

cat_dataset = CatDataset(file_paths=["./torch_transforms/cat.jpeg"],transform=torchvision_transform)

alb_oneof = alb_cat_dataset(file_paths=['./torch_transforms/cat.jpeg'], transform=albumentations_transform_oneof)

total_time = 0
for i in range(100):
    image, end_time = cat_dataset[0]
    total_time += end_time

print("torchvision time/image >> ", total_time*10)
# torchvision time/image >>  13.368799686431885
# torchvision time/image >>  14.29417610168457

# plt.figure(figsize=(10,10))
# plt.imshow(transforms.ToPILImage()(image))
# plt.show()

alb_total_time = 0
for i in range(100):
    alb_image, alb_time = alb_dataset[0]
    alb_total_time += alb_time

print("alb time >> ", alb_total_time*10)
# alb time >>  1.891648769378662

# torchvision time/image >>  14.29417610168457
# alb time >>  1.891648769378662

# 선생님 코드 결과
# torchvision tiem/image >>  4.332616329193115
# alb time >>  1.908423900604248

alb_total_time1 = 0
for i in range(100):
    alb_image1, alb_time1 = alb_oneof[0]
    alb_total_time1 += alb_time1
print("alb_oneof time >> ", alb_total_time1*10)

# torchvision time/image >>  14.82564926147461
# alb time >>  2.3098325729370117
# alb_oneof time >>  6.339056491851807

plt.figure(figsize=(10,10))
plt.imshow(transforms.ToPILImage()(alb_image1))
plt.show()