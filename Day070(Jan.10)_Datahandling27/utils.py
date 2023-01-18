import os
import torch
import pandas as pd


# train loop
def train(num_epoch, model, train_loader, val_loader, criterion, optimizer, scheduler,save_dir, device):
    print("Start training.....")
    dfForAccuracy = pd.DataFrame(index=list(range(num_epoch)),
                                columns=["Epoch", "Accuracy"])

    total = 0
    best_loss = 777

    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device), labels.to(device)
            output = model(img)

            train_loss = criterion(output, label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)

            train_acc = (label == argmax).float().mean()

            total += label.size(0)

            if (i + 1) % 10 == 0:
                print("Epoch >> [{}/{}], step >> [{}/{}], Loss >> {:.4f}, train_acc >> {:.2f}%"
                      .format(epoch + 1, num_epoch, i + 1, len(train_loader), train_loss.item(), train_acc.item() * 100))

        avrg_loss, val_acc = validation(model, val_loader, criterion, device)

        if avrg_loss < best_loss:
            print("Best pt save")
            best_loss = avrg_loss
            save_model(model, save_dir)

        if scheduler is not None:
            scheduler.step(avrg_loss)
        dfForAccuracy.loc[epoch, "Epoch"] = epoch + 1
        dfForAccuracy.loc[epoch, 'Train_Acc'] = round(train_acc.item(), 3)
        dfForAccuracy.loc[epoch, 'Val_Acc'] = round(val_acc, 3)        
        dfForAccuracy.loc[epoch, 'Train_Loss'] = round(train_loss.item(), 3)
        dfForAccuracy.loc[epoch, 'Val_Loss'] = round(avrg_loss, 3)

        dfForAccuracy.to_csv("./modelAccuracy.csv", index=False)

    save_model(model, save_dir, file_name="last.pt")

def validation(model, val_loader, criterion, device):
    print("val Start !!! ")
    model.eval() 

    with torch.no_grad():
        total = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        correct = 0

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
        
    avrg_loss = total_loss / cnt 
    val_acc = (correct / total * 100)
    print("Acc >> {:.2f} Average loss >> {:.4f}".format(
        val_acc, avrg_loss
    ))

    model.train()

    return avrg_loss, val_acc


def save_model(model, save_dir, file_name = "resnet18.pt"): # default는 best.pt로 해둔다.
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path) 
