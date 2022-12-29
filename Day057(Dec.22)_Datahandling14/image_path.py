## 이미지 경로를 입력 받아서 처리하는 코드

import os
import glob
import argparse     # 명령의 옵션, 커맨드라인에서 경로 인자를 입력 받는 기능(파이썬 문법 - 모듈부분)
# python main.py --data ./dataset/
from PIL import Image

# image_pretreatment.py + 부분

# 오렌지 : Orange
# 자몽 : Grapefruit
# 레드향 : Kanpei
# 한라봉 : Dekopon

# 폴더 구성 /dataset/image/각폴더명 생성/ 이미지 저장까지(resize 400x400)
# 직사각형 -> 정사각형 리사이즈 비율 유지하는 함수

def expand2square(img, backgroundcolor):
    width_temp, height_temp = img.size
    if width_temp == height_temp:
        return img
    elif width_temp > height_temp:
        result = Image.new(img.mode, (width_temp, width_temp), backgroundcolor)
        result.paste(img, (0,(width_temp-height_temp)//2))
        return result
    elif width_temp < height_temp:
        result = Image.new(img.mode, (height_temp, height_temp), backgroundcolor)
        result.paste(img, ((height_temp -width_temp)//2, 0))
        return result

    


def image_processing(image_orange, image_grapefruit, image_kanpei, image_dekopon):
    orange = image_orange
    grapefruit = image_grapefruit
    kenpei = image_kanpei
    dekopon = image_dekopon

    for i in orange:
        # 이미지 읽고 가로, 세로를 expand2square()에 던지도록
        # print(i)
        file_name = i.split('\\')
        # print(file_name) # ['./image', 'orange', '90.jpg']
        file_name = file_name[2]
        file_name = file_name.replace('.jpg','.png')
        orange_img = Image.open(i)
        orange_img_resize = expand2square(orange_img, (0,0,0)).resize((400,400))
        
        # 폴더 생성
        os.makedirs("./dataset/image/orange", exist_ok=True)
        orange_img_resize.save(f"./dataset/image/orange/orange_{file_name}")

    for i in grapefruit:
        file_name = i.split('\\')
        file_name = file_name[2]
        file_name = file_name.replace('.jpg','.png')
        grapefruit_img = Image.open(i)
        grapefruit_img_resize = expand2square(grapefruit_img, (0,0,0)).resize((400,400))
        
        # 폴더 생성
        os.makedirs("./dataset/image/grapefruit", exist_ok=True)
        grapefruit_img_resize.save(f"./dataset/image/grapefruit/grapefruit_{file_name}")

    for i in kenpei:
        file_name = i.split('\\')
        file_name = file_name[2]
        file_name = file_name.replace('.jpg','.png')
        kenpei_img = Image.open(i)
        kenpei_img_resize = expand2square(kenpei_img, (0,0,0)).resize((400,400))
        
        # 폴더 생성
        os.makedirs("./dataset/image/kenpei", exist_ok=True)
        kenpei_img_resize.save(f"./dataset/image/kenpei/kenpei_{file_name}")

    for i in dekopon:
        file_name = i.split('\\')
        file_name = file_name[2]
        file_name = file_name.replace('.jpg','.png')
        dekopon_img = Image.open(i)
        dekopon_img_resize = expand2square(dekopon_img, (0,0,0)).resize((400,400))
        
        # 폴더 생성
        os.makedirs("./dataset/image/dekopon", exist_ok=True)
        dekopon_img_resize.save(f"./dataset/image/dekopon/dekopon_{file_name}")


def image_file_check(opt):
    # "--image-folder-path"
    image_path = opt.image_folder_path
    # print(image_path)

    # 각 폴더별 데이터 양 체크
    all_image = glob.glob(os.path.join(image_path,'*','*.jpg'))
    # print("all_images >>",len(all_image))    

    # 오렌지
    image_orange = glob.glob(os.path.join(image_path,'orange','*.jpg'))
    # print("orange images >>",len(image_orange))
    # 자몽
    image_grapefruit = glob.glob(os.path.join(image_path,'grapefruit','*.jpg'))
    # print("grapefruit images >>",len(image_grapefruit))
    # 레드향
    image_kanpei = glob.glob(os.path.join(image_path,'kanpei','*.jpg'))
    # print("kanpei images >>",len(image_kanpei))
    # 한라봉
    image_dekopon = glob.glob(os.path.join(image_path,'dekopon','*.jpg'))
    # print("dekopon images >>",len(image_dekopon))

    return image_orange, image_grapefruit, image_kanpei, image_dekopon
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder-path", type=str, default="./image")
    opt = parser.parse_args()

    return opt


if __name__ =="__main__":
    opt = parse_opt()
    image_orange, image_grapefruit, image_kanpei, image_dekopon = image_file_check(opt)
    image_processing(image_orange, image_grapefruit, image_kanpei, image_dekopon)