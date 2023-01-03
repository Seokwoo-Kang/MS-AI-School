import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import timm
from timm.loss import LabelSmoothingCrossEntropy
from custom_dataset import custom_dataset

# pip install timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device) -->> cuda

# dataset
train_data = custom_dataset("D:\\data\\train")
test_data = custom_dataset("D:\\data\\test")

# for i in test_data:
#     print(i)

# dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# model call
net = models.__dict__["resnet18"](pretrained=True)
net.fc = nn.Linear(512, 10)
net.to(device)
# print(net)

# lossfunction
criterion = LabelSmoothingCrossEntropy()
# this is better than nn.CrossEntropyLoss   라벨이 명확하게 주어지지 않기때문에 의문점을 가지게 해서 학습시킴
criterion = criterion.to(device)

# optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

net.train()
total_step = len(train_loader)
curr_lr = 0.001
best_score = 0
num_epochs = 100

for epoch in range(num_epochs+1):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1)%100==0:
            print("{}/{}". format(12*(i+1), train_data.__len__()))

    net.eval()
    score = 0
    total =0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)

        total += images.size(0)
        _, argmax = torch.max(output, 1)
        score += (labels==argmax).sum().item()
    print("Epoch : {}, Loss :{:.4f}". format(
        epoch+1, total_loss/total_step
    ))

    avg = (score / total*100)
    print("Accuracy : {:.2f}\n".format(avg))
    net.train()

    if best_score < avg:
        best_score = avg
        torch.save(net.state_dict(), "./best.pt")