import torchvision.models as models

model = models.resnet18(pretrained=True)
print(model)
