import os
import glob
from sklearn.model_selection import train_test_split
import cv2
import natsort

# 데이터 나누기
# 보통 train 80, val 20, test 10
# 데이터가 적기 때문에 train 90, val 10으로

"""
data
    - train
        -dekopon
            -dekopon_0000.png
        -grapefruit
        -kenpei
        -orange
    - val
        -dekopon
            -dekopon_0000.png
        -grapefruit
        -kenpei
        -orange
    - test
        -dekopon
            -dekopon_0000.png
        -grapefruit
        -kenpei
        -orange
"""

# 경로 잡고 저장

orange_image_path = "./dataset/image/orange/"
dekopon_image_path = "./dataset/image/dekopon/"
grapefruit_image_path = "./dataset/image/grapefruit/"
kenpei_image_path = "./dataset/image/kenpei/"

orange_image_full_path = natsort.natsorted(glob.glob(os.path.join(f"{orange_image_path}/*.png")))
dekopon_image_full_path = natsort.natsorted(glob.glob(os.path.join(f"{dekopon_image_path}/*.png")))
grapefruit_image_full_path = natsort.natsorted(glob.glob(os.path.join(f"{grapefruit_image_path}/*.png")))
kenpei_image_full_path = natsort.natsorted(glob.glob(os.path.join(f"{kenpei_image_path}/*.png")))

# split

orange_train_data, orange_val_data = train_test_split(orange_image_full_path, test_size=0.1, random_state=7777)

dekopon_train_data, dekopon_val_data = train_test_split(dekopon_image_full_path, test_size=0.1, random_state=7777)

grapefruit_train_data, grapefruit_val_data = train_test_split(grapefruit_image_full_path, test_size=0.1, random_state=7777)

kenpei_train_data, kenpei_val_data = train_test_split(kenpei_image_full_path, test_size=0.1, random_state=7777)

# orange
for orange_train_data_path in orange_train_data:
    orange_train_img = cv2.imread(orange_train_data_path)
    orange_train_file_name = os.path.basename(orange_train_data_path)
    os.makedirs("./dataset/train/orange/", exist_ok=True)
    cv2.imwrite(f"./dataset/train/orange/{orange_train_file_name}", orange_train_img)

for orange_val_path in orange_val_data:
    orange_val_img = cv2.imread(orange_val_path)
    orange_val_name = os.path.basename(orange_val_path)
    os.makedirs("./dataset/val/orange/", exist_ok=True)
    cv2.imwrite(f"./dataset/val/orange/{orange_val_name}", orange_val_img)

# dekopon
for dekopon_train_data_path in dekopon_train_data:
    dekopon_train_img = cv2.imread(dekopon_train_data_path)
    dekopon_train_file_name = os.path.basename(dekopon_train_data_path)
    os.makedirs("./dataset/train/dekopon/", exist_ok=True)
    cv2.imwrite(f"./dataset/train/dekopon/{dekopon_train_file_name}", dekopon_train_img)

for dekopon_val_path in dekopon_val_data:
    dekopon_val_img = cv2.imread(dekopon_val_path)
    dekopon_val_name = os.path.basename(dekopon_val_path)
    os.makedirs("./dataset/val/dekopon/", exist_ok=True)
    cv2.imwrite(f"./dataset/val/dekopon/{dekopon_val_name}", dekopon_val_img)

# grapefruit
for grapefruit_train_data_path in grapefruit_train_data:
    grapefruit_train_img = cv2.imread(grapefruit_train_data_path)
    grapefruit_train_file_name = os.path.basename(grapefruit_train_data_path)
    os.makedirs("./dataset/train/grapefruit/", exist_ok=True)
    cv2.imwrite(f"./dataset/train/grapefruit/{grapefruit_train_file_name}", grapefruit_train_img)

for grapefruit_val_path in grapefruit_val_data:
    grapefruit_val_img = cv2.imread(grapefruit_val_path)
    grapefruit_val_name = os.path.basename(grapefruit_val_path)
    os.makedirs("./dataset/val/grapefruit/", exist_ok=True)
    cv2.imwrite(f"./dataset/val/grapefruit/{grapefruit_val_name}", grapefruit_val_img)

# kenpei
for kenpei_train_data_path in kenpei_train_data:
    kenpei_train_img = cv2.imread(kenpei_train_data_path)
    kenpei_train_file_name = os.path.basename(kenpei_train_data_path)
    os.makedirs("./dataset/train/kenpei/", exist_ok=True)
    cv2.imwrite(f"./dataset/train/kenpei/{kenpei_train_file_name}", kenpei_train_img)

for kenpei_val_path in kenpei_val_data:
    kenpei_val_img = cv2.imread(kenpei_val_path)
    kenpei_val_name = os.path.basename(kenpei_val_path)
    os.makedirs("./dataset/val/kenpei/", exist_ok=True)
    cv2.imwrite(f"./dataset/val/kenpei/{kenpei_val_name}", kenpei_val_img)


