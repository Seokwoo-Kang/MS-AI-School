import rexnetv1
from customdataset import customDataset
import torch
import torch.optim as optim
from rexnetv1 import ReXNetV1
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy # <- 이 로스도 추가시켰다.
from utils import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.RandomShadow(p=0.5),
    A.RandomFog(p=0.4),
    A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(25, p=0.7),
    A.ShiftScaleRotate(shift_limit=5, scale_limit=0.05,
                       rotate_limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# data set
train_dataset = customDataset(".\\dataset\\train\\", transform=train_transform)
val_dataset = customDataset(".\\dataset\\val\\", transform=val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4) # .num_workers=4를 써주면 사용할 수 있다.
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

# Pretrained start (Pretrained를 사용하는 경우)
model = rexnetv1.ReXNetV1()
model.load_state_dict(torch.load("./rexnetv1_1.0.pth"))
model.output[1] = nn.Conv2d(1280, 50, kernel_size=1, stride=1)
model.to(device)
# print(model)

# model.load_state_dict(torch.load("./best.pt") 이걸로 바꾸어주면 학습된걸로 할 수 있다.


# Pretrained no start (Pretrained를 사용하지 않는 경우)
# model = rexnetv1.ReXNetV1(classes=50)
# model.to(device)

# loss하고 옵티를 구성해준다.
criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.001) # 모델의 파라메터를 준다. 그리고 0.001이 기본이다.
save_dir = "./"
num_epoch = 100

# 이건 train의 값이 뭐인지 보는 것 train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device)
### train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device)

# num_workers를 쓰면 아래 코드를 작성해주어야 한다.
if __name__ == "__main__":
    train(num_epoch, model, train_loader, val_loader, criterion, optimizer, save_dir, device)

