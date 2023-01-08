from customdataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.optim as optim

from utils import train, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aug
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train val test dataset
# train val test loader

train_path = "./dataset/train"
val_path = "./dataset/val"
test_path = "./dataset/test"

train_dataset = CustomDataset(train_path, transform=train_transform)
val_dataset = CustomDataset(val_path, transform=val_transform)
test_dataset = CustomDataset(test_path, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# for i, (img, label) in enumerate(train_dataloader):
#     print(img, label)
#     exit()

# model 처리 완료(Train)
net = models.resnet18(pretrained=True)
in_feature_val = net.fc.in_features
# print(in_feature_val) # >>> 512
net.fc = nn.Linear(in_feature_val,4)
# print(net)
net.to(device)

# # model loader(Test)
# net = models.resnet18(pretrained=False)
# net.load_state_dict(torch.load("./best.pt", map_location=device))


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
if __name__=='__main__':
    train(100, train_dataloader, val_dataloader, net, optimizer, criterion, device, save_path="./best.pt")
    # test(net, test_loader, device)