from customdataset import customDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd

def test_result( model, val_loader, criterion, device):
    print("test Start !!! ")
    model.eval() # 평가 모델로 전환해준다.
    dfForAccuracy = pd.DataFrame(index=list(range(len(val_loader))),
                                columns=["Epoch", "Accurascy"])

    with torch.no_grad():
        total = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        correct = 0
        # for epoch in range(num_epoch):    
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss.item()
            cnt += 1
        
        avrg_loss = total_loss / cnt # 평균 로스가 나오게 된다.
        val_acc = (correct / total * 100)
        print("Acc >> {:.2f} Average loss >> {:.4f}".format(
            val_acc, avrg_loss
        ))
        dfForAccuracy.loc[i, "Epoch"] = i + 1
        dfForAccuracy.loc[i, 'Test_Acc'] = round(val_acc, 3)
        dfForAccuracy.loc[i, 'Test_Loss'] = round(avrg_loss, 3)

        dfForAccuracy.to_csv("./resnet34_test2.csv", index=False)
        model.train()

    return avrg_loss, val_acc




