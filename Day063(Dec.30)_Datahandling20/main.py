import argparse
import copy

import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from customdataset import customDataset
from timm.loss import LabelSmoothingCrossEntropy
from utils import train
# pip install adamp
# from adamp import AdamP

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




def main(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    # augmentations
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,rotate_limit=15, p=0.8),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        # A.Normalize(mean=(0.5,0.5,0.5), std=(0.2,0.2,0.2))
        A.RandomShadow(p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
        ])
    val_transform = A.Compose([
        A.SmallestMaxSize(max_size=160),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
        ])

    # dataset, dataloader
    train_dataset = customDataset(img_path=opt.train_path, transform=train_transform)
    val_dataset = customDataset(img_path=opt.val_path, transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    #model call
    net = models.__dict__["resnet50"](pretrained=True)
    net.fc = nn.Linear(512, 53)
    net.to(device)
    print(net)

    # loss
    criterion = LabelSmoothingCrossEntropy().to(device)

    # optimizer
    #(self, params: _params_t, lr: float=..., betas: Tuple[float, float]=..., eps: float=..., weight_decay: float=..., amsgrad: bool = ...
    optimizer = optim.AdamW(net.params(),lr=opt.lr,weight_decay=1e-2)

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,90],gamma=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # model.pt save dir
    save_dir = opt.save_path
    os.makedirs(save_dir, exist_ok=True)
    # train(num_epoch, model, train_loader, val_loader,
    # criterion,optimizer, scheduler, save_dir, device):
    train(opt.epoch, net, train_loader, val_loader, criterion, optimizer,
          scheduler, save_dir, device)

    # train -> label ->
# 직접 입력을 받도록
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./dataset/train", help="train data path")
    parser.add_argument("--val-path", type=str, default="./dataset/valid", help="val data path")
    parser.add_argument("--batch-size", type=int, default=32,help="batch size")
    parser.add_argument("--epoch", type=int, default=100,help="epoch number")
    parser.add_argument("--lr", type=float, default=0.001,help="lr number")
    parser.add_argument("--save-path", type=str, default="./weights",help="save model")
    opt =parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    # # 텐서형태로 바꾼 데이터를 이미지로 보기위해 다시 역과정
    # def visualize_aug(dataset, idx=0, samples=10, cols=5):
    #     dataset = copy.deepcopy(dataset)
    #     dataset.transform = A.Compose([
    #         t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    #     ])
    #     rows = samples//cols
    #     figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))
    #
    #     for i in range(samples):
    #         image, _ = dataset[idx]
    #         ax.ravel()[i].imshow(image)
    #         ax.ravel()[i].set_axis_off()
    #     plt.tight_layout()
    #     plt.show()
    #
    # visualize_aug(train_dataset)