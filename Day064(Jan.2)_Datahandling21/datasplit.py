import os
import glob
from sklearn.model_selection import  train_test_split
from torch.utils.data import Dataset
import cv2

# data load acn split
# train, val, test


# img_folder_path = "./data"
# img_folder = glob.glob(os.path.join(img_folder_path,"*", "*.jpg"))
# train_list, val_list = train_test_split(img_folder, test_size=0.2, random_state=7777)
# val_data, test_data = train_test_split(val_list, test_size=0.5, random_state=7777)
# print(len(train_list), len(val_data), len(test_data))
# # 4504 563 564
# # 이렇게 하면 안된다. 이러면 라벨별 비율이 맞지 않을 가능성이 있음

img_cloudy_path = "./data/cloudy"
img_cloudy = glob.glob(os.path.join(img_cloudy_path, "*.jpg"))
img_desert_path = "./data/desert"
img_desert = glob.glob(os.path.join(img_desert_path, "*.jpg"))
img_green_area_path = "./data/green_area"
img_green_area = glob.glob(os.path.join(img_green_area_path, "*.jpg"))
img_water_path = "./data/water"
img_water = glob.glob(os.path.join(img_water_path, "*.jpg"))
# print(len(img_cloudy),len(img_desert),len(img_green_area),len(img_water))
# 1500 1131 1500 1500

cloudy_tr_list, cloudy_val_list = train_test_split(img_cloudy, test_size=0.2,random_state=7777)
cloudy_val_data, cloudy_test_data = train_test_split(cloudy_val_list, test_size=0.5, random_state=7777)
desert_tr_list, desert_val_list = train_test_split(img_desert, test_size=0.2,random_state=7777)
desert_val_data, desert_test_data = train_test_split(desert_val_list, test_size=0.5, random_state=7777)
green_area_tr_list, green_area_val_list = train_test_split(img_green_area, test_size=0.2,random_state=7777)
green_area_val_data, green_area_test_data = train_test_split(green_area_val_list, test_size=0.5, random_state=7777)
water_tr_list, water_val_list = train_test_split(img_water, test_size=0.2,random_state=7777)
water_val_data, water_test_data = train_test_split(water_val_list, test_size=0.5, random_state=7777)
# print(len(cloudy_tr_list),len(cloudy_val_data),len(cloudy_test_data))   #>>> "1200 150 150"
# print(len(desert_tr_list),len(desert_val_data),len(desert_test_data))   #>>> "904 113 114"
# print(len(green_area_tr_list),len(green_area_val_data),len(green_area_test_data))   #>>> "1200 150 150"
# print(len(water_tr_list),len(water_val_data),len(water_test_data))  #>>> "1200 150 150"

def data_save(data, mode):
    for path in data:
        # image name
        image_name = os.path.basename(path)
        image_name = image_name.replace(".jpg", "")
        # print(image_name)
        # # train_17357

        # 0. 폴더 명 구하기
        folder_path = path.split("\\")
        folder_name = folder_path[0].split("/")
        folder_name = folder_name[2]
        # print(folder_name)  # ['.', 'data', 'cloudy\\train_17357.jpg']
        # print(folder_name[2])  # cloudy

        # 1. 폴더 구성
        # move or image save
        # ./data/cloudy\train_17357.jpg
        folder = f"./dataset/{mode}/{folder_name}"
        os.makedirs(folder, exist_ok=True)

        # 2. 이미지 읽기
        img = cv2.imread(path)

        # 3. 이미지 저장
        # print(os.path.join(folder, image_name+".png"))
        # # ./dataset/train/cloudy\train_17357.png
        cv2.imwrite(os.path.join(folder, image_name+".png"), img)


data_save(water_test_data, mode="test")