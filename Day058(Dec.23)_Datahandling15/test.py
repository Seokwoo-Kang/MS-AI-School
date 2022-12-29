import torch
from torchvision.transforms import transforms
from torchvision import models
from dataset_custom import custom_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2

def acc_function(correct, total):
    acc = correct /total * 100
    return acc

def test(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad() :
        for batch_idx, (data, target, path) in enumerate(test_loader):
            image_path = path[0]
            img = cv2.imread(image_path)
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, argmax = torch.max(output, 1)
            total == target.size(0)
            correct += (target ==argmax).sum().item()
            font_italic = "FONT_ITALIC"

            argmax_temp = argmax.item()
            cv2.putText(img, str(argmax_temp), (90,90), cv2.FONT_ITALIC, 1, (0,0,0), 2)
            cv2.imshow("Test", img)
            cv2.waitKey(0)            

        acc=acc_function(correct, total)
        print("acc for {} image : {:.2f}%".format(total, acc))
        # print("Testing Accuracy: {:.2%}".format(acc))

def main() :
    test_transform = A.Compose([
        A.Resize(height=224,width=224),
        ToTensorV2()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.__dict__["resnet18"](pretrained=False, num_classes=4)
    net = net.to(device)

    net.load_state_dict(torch.load("./model_save/final.pt", map_location=device))
    test_data = custom_dataset("./data/val", transform=test_transform)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 데이터을 잘 받아오는지 확인
    # for i in test_data:
    #     print(i)
    # for i,j in test_loader:
    #     print(i,j)



# 디버깅
if __name__ =="__main__":
    main()

    

