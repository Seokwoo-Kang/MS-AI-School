from sklearn.model_selection import train_test_split
import os
import glob
import shutil

# 경로

image_path = "./dataset/image"

# 이미지 경로 -> list
dekopon_data = glob.glob(os.path.join(image_path, "dekopon", "*.png"))
orange_data = glob.glob(os.path.join(image_path, "orange", "*.png"))
grapefruit_data = glob.glob(os.path.join(image_path, "grapefruit", "*.png"))
kenpei_data = glob.glob(os.path.join(image_path, "kenpei", "*.png"))

# print(orange_data)

# split

orange_train_data, orange_val_data = train_test_split(orange_data, test_size=0.1, random_state=7777)

dekopon_train_data, dekopon_val_data = train_test_split(dekopon_data, test_size=0.1, random_state=7777)

grapefruit_train_data, grapefruit_val_data = train_test_split(grapefruit_data, test_size=0.1, random_state=7777)

kenpei_train_data, kenpei_val_data = train_test_split(kenpei_data, test_size=0.1, random_state=7777)

# dekopon
for i in dekopon_train_data:
    # ./dataset/image/dekopon/dekopon_105.png
    # file_name = i.split('\\')
    file_name = os.path.basename(i)
    print(file_name)
    os.makedirs("./data/train/dekopon/", exist_ok=True)
    shutil.move(i, f"./data/train/dekopon/{file_name}")

for i in dekopon_val_data:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/dekopon/", exist_ok=True)
    shutil.move(i, f"./data/val/dekopon/{file_name}")

# orange    
for i in orange_train_data:
    # ./dataset/image/dekopon/dekopon_105.png
    # file_name = i.split('\\')
    file_name = os.path.basename(i)
    print(file_name)
    os.makedirs("./data/train/orange/", exist_ok=True)
    shutil.move(i, f"./data/train/orange/{file_name}")

for i in orange_val_data:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/orange/", exist_ok=True)
    shutil.move(i, f"./data/val/orange/{file_name}")

# grapefruit
for i in grapefruit_train_data:
    # ./dataset/image/dekopon/dekopon_105.png
    # file_name = i.split('\\')
    file_name = os.path.basename(i)
    print(file_name)
    os.makedirs("./data/train/grapefruit/", exist_ok=True)
    shutil.move(i, f"./data/train/grapefruit/{file_name}")

for i in grapefruit_val_data:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/grapefruit/", exist_ok=True)
    shutil.move(i, f"./data/val/grapefruit/{file_name}")

# kenpei
for i in kenpei_train_data:
    # ./dataset/image/dekopon/dekopon_105.png
    # file_name = i.split('\\')
    file_name = os.path.basename(i)
    print(file_name)
    os.makedirs("./data/train/kenpei/", exist_ok=True)
    shutil.move(i, f"./data/train/kenpei/{file_name}")

for i in kenpei_val_data:
    file_name = os.path.basename(i)
    os.makedirs("./data/val/kenpei/", exist_ok=True)
    shutil.move(i, f"./data/val/kenpei/{file_name}")




