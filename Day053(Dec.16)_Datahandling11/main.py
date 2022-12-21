import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import models
import torch.nn as nn
from tqdm import tqdm

# import defaultdict 사전에 기본값이 있는 딕셔너리

# os.environ['KMP_DUPLICATE_LI_OK']='True'

# # m1 제품 해당하는것
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
# print('device', device)
 
# print(f"PyTorch version:{torch.__version__}")  # 1.12.1 이상 True 여야 합니다.
# print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}")
# print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}")  # True 여야 합니다.

# # 엔비디아 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

class catvsdogDataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        self.image_file_paths = glob.glob(os.path.join(image_file_path, "*","*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        
        # image loader
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)
        # convert BGR2RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image_path)

        # label
        label_temp = image_path.split("\\")
        # print(label_temp)
        # ['./Agumentation/dataset/dataset/train', 'dog', 'dog.99.jpg'] index = 1 확인
        label_temp = label_temp[1]
        label = 0
        if "cat" == label_temp:
            label = 0
        elif "dog" == label_temp:
            label == 1
        # print(label_temp, label)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    def __len__(self):
        return len(self.image_file_paths)

# agumentation
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=224, width=224),
    ToTensorV2()    
])

train_dataset = catvsdogDataset("./Agumentation/dataset/dataset/train", transform=train_transform)
val_dataset = catvsdogDataset("./Agumentation/dataset/dataset/val", transform=val_transform)

# visualize augmentation
def visualize_augmentation(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (ToTensorV2))])
    rows = samples//cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))
    for i in range(samples):
        image, _ =dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

# visualize_augmentation(train_dataset)

# 평가

def calculate_accuracy(output, target):
    output = target.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target==output).sum(dim=0), output.size(0)).item()
    # tensor를 정수 값으로 뽑고 싶을때 a.item() 사용

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val":0, "count":0, "avg":0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return "|".join(
            [
                "{metric_name} : {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

params = {
    "model" : "resnet18",
    "device" : "cpu",       # 본인 환경에 맞는 device설정 print(device)-->cpu
    "lr" : 0.001,
    "batch_size" : 64,
    # "num_workers" : 4,
    "epoch" : 10        # 원래 100번이상    
}

# model loader

# pretrained=True면 num_calsses를 1000으로 사용
model = models.__dict__[params["model"]](pretrained=True)
model.fc = nn.Linear(512,2)
model = model.to(params["device"])
# print(model)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])


# data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=params["batch_size"],
    shuffle=True
    # num_workers=params["num_workers"]
)
val_loader = DataLoader(
    val_dataset,
    batch_size=params["batch_size"],
    shuffle=False
    # num_workers=params["num_workers"]
)


# save model
def save_model(model, save_dir, file_name='last.pt'):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    # ./weight/last.pt 로 저장되도록
    if isinstance(model, nn.DataParallel):
        print("multi GPU activate")
        torch.save(model.module.state_dict(), output_path)
    else:
        print("single GPU activate")
        torch.save(model.state_dict(), output_path)


# train loop
def train(train_loader, model, criterion, optimizer, epoch, params, save_dir):
    metric_monitor = MetricMonitor()
    model.train()
    # pip install tqdm
    # import tqdm
    stream = tqdm(train_loader)
    for i, (image, target) in enumerate(train_loader):
        images = image.to(params["device"])
        targets = target.to(params["device"])

        output = model(images)
        loss = criterion(output, targets)
        accuracy = calculate_accuracy(output, targets)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch : {epoch}. Train.    {metric_monitor}".format(
                epoch= epoch, metric_monitor=metric_monitor
            )
        )

    save_model(model, save_dir)

# val loop
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():   # 기울기 구하지않고 학습하지 않겠다 -> 평가
        for i, (image, target) in enumerate(val_loader):
            images = image.to(params["device"])
            targets = target.to(params["device"])

            output = model(images)
            loss = criterion(output, targets)
            accuracy = calculate_accuracy(output, targets)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
                "Epoch : {epoch}. val.    {metric_monitor}".format(
                    epoch= epoch, metric_monitor=metric_monitor
                )
            )

save_dir = "./weight"


for epoch in range(1, params["epoch"] +1):
    train(train_loader, model, criterion, optimizer, epoch, params, save_dir)
    validate(val_loader, model, criterion, optimizer, epoch, params)
