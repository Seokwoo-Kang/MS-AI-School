
from customdataset import customDataset
import torch
import torch.optim as optim
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy # <- 이 로스도 추가시켰다.
from utils import train
from test import test_result
# from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_transform = A.Compose([
#     A.SmallestMaxSize(max_size=256),
#     A.Resize(height=224, width=224),
#     A.ToGray(p=1),
#     A.Rotate(60, p=1),
#     A.Rotate(45, p=1),
#     A.Rotate(30, p=1),
#     A.Rotate(15, p=1),
#     A.Rotate(75, p=1),
#     A.Rotate(90, p=1),
#     A.Rotate(105, p=1),
#     A.Rotate(120, p=1),
#     A.Rotate(135, p=1),
#     A.Rotate(150, p=1),
#     A.Rotate(165, p=1),
#     A.Rotate(180, p=1),
#     A.RandomShadow(p=0.4),
#     A.RandomFog(p=0.4),
#     A.RandomSnow(p=0.4),
#     A.RandomBrightnessContrast(p=0.4),
#     A.ShiftScaleRotate(shift_limit=5, scale_limit=0.05,
#                        rotate_limit=15, p=0.7),
#     A.VerticalFlip(p=0.6),
#     A.HorizontalFlip(p=0.6),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

# val_transform = A.Compose([
#     A.SmallestMaxSize(max_size=256),
#     A.Resize(height=224, width=224),
#     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

train_transform = A.Compose([
    A.SmallestMaxSize(max_size=160),
    A.Resize(height=224, width=224),
    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    # A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(15, p=1),
    A.Rotate(30, p=1),
    A.Rotate(45, p=1),
    A.Rotate(60, p=1),
    A.Rotate(75, p=1),
    A.Rotate(90, p=1),
    A.Rotate(105, p=1),
    A.Rotate(120, p=1),
    A.Rotate(135, p=1),
    A.Rotate(150, p=1),
    A.Rotate(165, p=1),
    A.Rotate(180, p=1),
    # A.ShiftScaleRotate(shift_limit=5, scale_limit=0.09, rotate_limit=25, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=160),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
test_transform = A.Compose([
    A.SmallestMaxSize(max_size=160),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# data set
train_dataset = customDataset(".\\dataset\\train\\", transform=train_transform)
val_dataset = customDataset(".\\dataset\\val\\", transform=val_transform)
test_dataset = customDataset(".\\dataset\\test\\", transform=val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True) # .num_workers=4를 써주면 사용할 수 있다.
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)

# Pretrained start (Pretrained를 사용하는 경우)
model = models.resnet34()
model.fc = nn.Linear(512, 3)
model.load_state_dict(torch.load("./models/best.pt"))
model.to(device)

# model.load_state_dict(torch.load("./best.pt") 이걸로 바꾸어주면 학습된걸로 할 수 있다.

# Pretrained no start (Pretrained를 사용하지 않는 경우)
# model = rexnetv1.ReXNetV1(classes=50)
# model.to(device)

# loss하고 옵티를 구성해준다.
criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.0001) # 모델의 파라메터를 준다. 그리고 0.001이 기본이다.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,threshold_mode='abs',min_lr=1e-9, verbose=True)
save_dir = "./models/"
num_epoch = 20

# 이건 train의 값이 뭐인지 보는 것 train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device)
### train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device)

# num_workers를 쓰면 아래 코드를 작성해주어야 한다.
if __name__ == "__main__":
    # train(num_epoch, model, train_loader, val_loader, criterion, optimizer,scheduler, save_dir, device)
    test_result(model, val_loader, criterion, device)
