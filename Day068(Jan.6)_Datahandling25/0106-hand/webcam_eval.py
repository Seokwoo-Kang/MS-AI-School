import numpy as np
import torch
import torch.nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import PIL
from torchvision import models
import torch.nn as nn
import cv2

# This is the Label
Labels = {0: 'paper',
          1: 'rock',
          2: 'scissors'
          }

# Let's preprocess the inputted frame

data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)

# data_transforms = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=160),
#         A.Resize(height=224, width=224),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2()
#     ]
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ##Assigning the Device which will do the calculation

# model  = torch.load("Resnet50_Left_Pretrained_ver1.1.pth") #Load model to CPU
# model = models.swin_t()
# model.head = nn.Linear(in_features=768, out_features=3)
# model.load_state_dict(torch.load("./best_swin_t.pt",map_location=torch.device('cpu')))

model = models.vit_b_16()
model.heads[0] = nn.Linear(in_features=768, out_features=3)
model.load_state_dict(torch.load("./best_vit_b_16.pt",map_location=torch.device('cpu')))
model = model.to(device)  # set where to run the model and matrix calculation
model.eval()  # set the device to eval() mode for testing


# Set the Webcam
def Webcam_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]

    return result, score


def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    print(image)
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.cpu()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Set the webcam
Webcam_720p()

fps = 0
show_score = 0
show_res = 'Nothing'
sequence = 0

while True:
    ret, frame = cap.read()  # Capture each frame

    if fps == 4: #
        # image = frame[100:450, 150:570]
        image = frame[400:900, 150:550]
        image_data = preprocess(image)
        print(image_data)
        prediction = model(image_data)
        result, score = argmax(prediction)
        score = float(score)
        fps = 0
        if score >= 0.7:
            show_res = result
            show_score = score
        else:
            show_res = "Nothing"
            show_score = score

    fps += 1
    # cv2.putText(frame, '%s' % (show_res), (950, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # cv2.putText(frame, '(score = %.5f)' % (show_score), (950, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
    cv2.putText(frame, '%s' % (show_res), (625, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(frame, '(score = %.5f)' % (show_score), (600, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
    cv2.imshow("ASL SIGN DETECTER", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("ASL SIGN DETECTER")