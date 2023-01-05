import torch
import copy
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# visualize_aug(train_dataset)
def visualize_aug(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    ])
    rows = samples//cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def train(num_epoch, model, train_loader, val_loader, criterion,optimizer, scheduler, save_dir, device):
    print("Start training......")
    running_loss = 0.0
    total = 0
    best_loss = 9999
    for epoch in range(num_epoch+1):
        for i, (imgs, labels) in enumerate(train_loader):
            img, label = imgs.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output, label)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()     # 시각화할때 사용

            _, argmax = torch.max(output, 1)        # argmax에 예측한 label이 저장
            acc = (label==argmax).float().mean()
            total += label.size(0)

            if (i+1)%10 ==0 :
                print("Epoch [{}/{}] Step[{}/{}] Loss :{:.4f} Acc:{:.2f}%".format(
                    epoch+1,num_epoch, i+1, len(train_loader), loss.item(), acc.item()*100
                ))

        avg_loss, val_acc = validation( model, val_loader, criterion, device)

        # if epoch % 10 == 0:
        #     save_model(model, save_dir, file_name=f"{epoch}.pt")

        if avg_loss < best_loss :
            print(f"Best model save epoch >> {epoch}")
            print("Save model in", save_dir)
            best_loss = avg_loss
            save_model(model, save_dir, file_name ="best_resnet.pt")

    save_model(model, save_dir, file_name ="last_resnet.pt")

def validation(model, val_loader, criterion, device):
    print("Start validation")

    with torch.no_grad():
        model.eval()

        total =0
        correct = 0
        total_loss =0
        cnt =0
        batch_loss =0

        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(output, 1)
            correct += (labels==argmax).sum().item()
            total_loss += loss
            cnt += 1

    avg_loss = total_loss/cnt
    val_acc = (correct/total)*100
    print("Val Acc : {:.2f}% Avg_loss : {:.4}".format(
        val_acc, avg_loss
    ))
    model.train()
    return avg_loss, val_acc

def save_model(model, save_dir, file_name="best_resnet.pt"):
    output_path = os.path.join(save_dir, file_name)

    torch.save(model.state_dict(), output_path)
