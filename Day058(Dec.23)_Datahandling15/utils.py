# train loop
# val loop
# save model
# 평가 함수
import torch
import os
import torch.nn as nn
from metric_monitor import MetricMonitor
from tqdm import tqdm


## nn?
## nn.


# # 평가 함수
# def calculate_acc(output, target):
#     output = torch.sigmoid(output) >= 0.5
#     target = target == 1.0

#     return torch.true_divide((output==target).sum(dim=0), output.size(0)).item()

# save model
def save_model(model, save_dir, file_name="last.pt"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    if isinstance(model, nn.DataParallel):
        print("멀티 GPU 저장 !!")
        torch.save(model.module.state_dict(), output_path)

    else:
        print("싱글 GPU 저장 !!")
        torch.save(model.state_dict(), output_path)


# 새로운 train loop
def train(number_epoch, train_loader, val_loader, criterion, optimizer, model, save_dir, device):
    print("start training...")
    running_loss = 0.0
    total = 0
    best_loss = 77777

    for epoch in range(number_epoch):
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, argmax = torch.max(outputs, 1)
            acc = (labels ==argmax).float().mean()
            total += labels.size(0)

            if ( i + 1 ) % 5 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}".format(
                    epoch + 1,
                    number_epoch,
                    i+1,
                    len(train_loader),
                    loss.item(),
                    acc.item()*100,
                ))
        
        avg_loss, val_acc = validate(epoch, model, val_loader, criterion, device)

        # 특정 epoch마다 저장하고 싶은 경우
        if epoch % 10 ==0:
            save_model(model, save_dir, file_name=f"{epoch}.pt")

        # best save
        if val_acc > best_loss:
            print(f"best save >>> {epoch}")
            best_loss = val_acc
            save_model(model, save_dir, file_name="best.pt")

    # last save
    save_model(model, save_dir, file_name="final.pt")


def validate(epoch, model, val_loader, criterion, device):
    print("start validation...")
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        cnt=0
        batch_loss = 0

        for i, (images, labels) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += labels.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (argmax ==labels).sum().item()
            total_loss += loss.item()
            cnt += 1

    avg_loss = total_loss /cnt
    val_acc = (correct/total*100)

    print("val # {} acc {:.2f}% avg loss {:.4f}".format(
        epoch + 1,
        correct/total *100,
        avg_loss,


    ))

    model.train()

    return avg_loss, val_acc













# # train loop
# def train(train_loader, model, criterion, optimizer, epoch, device):
#     metric_monitor = MetricMonitor()
#     model.train()
#     stream = tqdm(train_loader)
#     for batch_idx, (image, target) in enumerate(train_loader):
#         images = image.to(device)
#         """
#         print(images) =
#         [0,0,0,0,
#         0,0,0,0,
#         0,0,0,0].("cuda")
#         """
#         target = target.to(device)
#         output = model(images)  # output에 예측한 결과치가 저장
#         loss = criterion(output, target)
#         accuracy = calculate_acc(output, target)

#         metric_monitor.update("Loss", loss.item())
#         metric_monitor.update("Accuracy", accuracy.item())
#         optimizer.zero_grad()
#         loss.backword()
#         optimizer.step()

#         stream.set_description(
#             f"Epoch : {epoch}.  Train...{metric_monitor}".format(
#                 epoch=epoch, metric_monitor=metric_monitor
#             )
#         )


# # val loop
# def validate(val_loader, model, criterion ,epoch, device):
#     metric_monitor = MetricMonitor()
#     model.eval()
#     stream = tqdm(val_loader)
#     for batch_idx, (image, target) in enumerate(val_loader):
#         images = image.to(device)
#         target = target.to(device)
#         output = model(images)
#         loss = criterion(output, target)
#         accuracy = calculate_acc(output, target)

#         metric_monitor.update("Loss", loss.item())
#         metric_monitor.update("Accuracy", accuracy.item())
#         stream.set_description(
#             f"Epoch:{epoch}. Val...{metric_monitor}".format(
#                 epoch=epoch, metric_monitor=metric_monitor
#             )
#         )

