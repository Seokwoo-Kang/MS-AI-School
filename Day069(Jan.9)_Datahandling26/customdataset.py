import os
import glob
import cv2
from torch.utils.data import Dataset

class Mycustomdataset(Dataset):
    def __init__(self, path_all, transform=None):
        self.file_path = glob.glob(os.path.join(path_all, "*", "*.png"))
        self.transform= transform
        self.label_dict={
            'r':0,
            's':1,
            'p':2
        }
        
    def __getitem__(self, item):
        image_path = self.file_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label = image_path.split('\\')[1]
        label = self.label_dict[label]
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
    
    def __len__(self):
        return len(self.file_path)

if __name__ == '__main__':
    train = Mycustomdataset('./dataset/train',transform=None)
    for i in train:
        # print(i)
        pass