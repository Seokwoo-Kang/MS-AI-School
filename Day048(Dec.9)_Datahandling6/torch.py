import torch
from torchvision import datasets
from torchvision.transform import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download =True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download =True,
    transform=ToTensor()
)