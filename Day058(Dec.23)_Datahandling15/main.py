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
train_dataset = custom_dataset("./project/data/train", transform=train_transform)
val_dataset = custom_dataset("./project/data/val", transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=hy_params.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hy_params.batch_size, shuffle=False)

# # model call
# net = models.__dict__["resnet18"](pretrained=False, num_classes=hy_params.num_classes)
# # print(net)
# pretrained = True, num_classes =4 수정방법
net = models.__dict__["resnet18"](pretrained=True)
net.fc = nn.Linear(512,4)
# print(net)
net.to(device)

# criterion
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=hy_params.lr)

# model save dir
model_save_dir="./model_save"
os.makedirs(model_save_dir, exist_ok=True)

# 새로운 train
# def train(number_epoch, train_loader, val_loader, criterion, optimizer, model, save_dir, device):
train(
    number_epoch=hy_params.epoch,
    train_loader=train_loader, 
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optim,
    model=net,
    save_dir=model_save_dir,
    device=device
    )


# # train
# for epochs in range(1, hy_params.epoch+1):
#     train(train_loader, net, criterion, optim, epochs, device)
#     validate(val_loader, net, criterion, epochs, device)

# # save
# model_path = "./"
# save_model(net, model_path, file_name="last.pt")




# # 디버깅

# for i, t in val_dataset:
#     # print(i)
#     # print(i, t)
#     print(type(i), t)

# for i, (image,target) in enumerate(train_loader):
#     print(i, image, target)

for i, (epoch, loss) in enumerate(train):
    print(epoch, loss)
