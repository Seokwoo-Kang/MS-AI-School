"""
폴더 구조
dataset
    - train
        - cat
            - cat1.jpg...
        - dog
    - val
        - cat
        - dog
    - test
        - cat
        - dog
"""

from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image


class cat_dog_mycustomdataset(Dataset):
    # csv folder 읽기, 변환 할당, 데이터 필터링 등과 같은 초기 논리가 발생
    def __init__(self, data_path):
        # data_path -> ./dataset/
        # train -> ./dataset/train
        # val -> ./dataset/val
        # test -> ./dataset/test
        self.all_data_path = glob.glob(os.path.join(data_path, '*', '*.jpg'))
        # -> dataset/train/cat or dog
        # print(self.all_data_path)

        pass
    
    # 데이터 레이블 반환 image, label
    def __getitem__(self, index):
        image_path = self.all_data_path[index]
        # print(image_path)
        img = Image.open(image_path).convert("RGB")
        label_temp = image_path.split("/")
        # print(label_temp)

        exit()


        return img        
        pass

    # 전체 데이터 길이 반환
    def __len__(self):

        pass


test = cat_dog_mycustomdataset("./dataset/train/")

for i in test:
    print(i)
    pass