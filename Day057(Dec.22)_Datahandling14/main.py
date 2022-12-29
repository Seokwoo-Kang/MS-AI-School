from dataset_custom import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch

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
train_dataset = custom_dataset("./data/train", transform=train_transform)
val_dataset = custom_dataset("./data/val", transform=val_transform)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)



# # 디버깅

# for i, t in val_dataset:
#     # print(i)
#     # print(i, t)
#     print(type(i), t)

for i, (image,target) in enumerate(train_loader):
    print(i, image, target)