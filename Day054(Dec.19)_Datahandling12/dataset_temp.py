import cv2
import glob
import os

from torch.utils.data import Dataset

label_dict = {"dekopon": 0,  "grapefruit": 1,  "kanpei": 2, "orange": 3}


class custom_dataset(Dataset):
    def __init__(self, image_file_path, transform=None):
        """
        data
            train
                라벨 폴더명 
                    이미지

        "./data/train"
        """
        self.image_file_paths = glob.glob(
            os.path.join(image_file_path, "*", "*.png"))
        self.transform = transform

    def __getitem__(self, index):
        # image loader
        image_path = self.image_file_paths[index]
        image = cv2.imread(image_path)
        print(image)

    def __len__(self):
        pass


if __name__ == '__main__':
    test = custom_dataset("./data/train", transform=None)
    for i in test:
        pass
