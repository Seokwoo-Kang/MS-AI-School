import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class MDC(Dataset):
    def __init__(self, path, transform=None):
        self.all_path = glob.glob(os.path.join(path, "*","*.png"))
        self.transform = transform
        self.label_dict = {"dark":0,"green":1, "light":2,"medium":3}

    def __getitem__(self, item):
        image_path = self.all_path[item]
        image = Image.open(image_path).convert('RGB')

        labels = image_path.split("\\")[1]
        label = self.label_dict[labels]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_path)