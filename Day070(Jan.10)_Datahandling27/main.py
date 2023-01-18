
from customdataset import MCD
import torch
import torch.optim as optim
import torchvision.models as models

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
from utils import train
import warnings
warnings.filterwarnings('ignore')

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

train_transform = A.Compose([
    A.SmallestMaxSize(max_size=160),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.09,
                        rotate_limit=25, p=0.6),
    A.Resize(width=224, height=224),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.6),
    A.RandomBrightnessContrast(p=0.6),
    A.HorizontalFlip(p=0.6),
    A.GaussNoise(p=0.5),
    A.Equalize(p=0.5),
    A.VerticalFlip(p=0.6),
    A.ISONoise(always_apply=False, p=0.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.22)),
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


train_dataset = MCD("./dataset/train/", transform=train_transform)
val_dataset = MCD("./dataset/val/", transform=val_transform)
test_dataset = MCD("./dataset/test/", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True) # .num_workers=4를 써주면 사용할 수 있다.
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)

# # Pretrained start (Pretrained를 사용하는 경우)
# model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(in_features=2048, out_features=10)
# model.load_state_dict(torch.load("./models/best.pt", map_location=device))
# model.to(device)
# # Pretrained no start (Pretrained를 사용하지 않는 경우)
# model = rexnetv1.ReXNetV1(classes=50)
# model.to(device)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=10)
# model.load_state_dict(torch.load("./models/best.pt", map_location=device))
model.to(device)


criterion = LabelSmoothingCrossEntropy()
optimizer = optim.Adam(model.parameters(), lr=0.01) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,threshold_mode='abs',min_lr=1e-9, verbose=True)
save_dir = "./models/"
num_epoch = 30

def test(model, test_loader, device) :
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad() :
        for i, (image, labels) in enumerate(test_loader) :
            image, labels = image.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output,1)
            total += image.size(0)
            correct += (labels == argmax).sum().item()

        acc = correct / total * 100
        print("acc for {} image : {:.2f}%".format(
            total, acc
        ))

if __name__ == "__main__":
    # train(num_epoch, model, train_loader, val_loader, criterion, optimizer,scheduler, save_dir, device)
    test(model, val_loader, device)
