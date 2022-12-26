import os
import glob
from PIL import Image

# image_pretreatment.py
image_path = "image"

# 오렌지 : Orange
# 자몽 : grapefruit
# 레드향 : Kanpei
# 한라봉 : Dekopon

# 폴더 구성 /dataset/image/각폴더명 생성/ 이미지 저장까지(resize 400x400)
# 직사각형 -> 정사각형 리사이즈 비율 유지하는 함수
# image_path.py에서


def image_file_check(image_path):

    # 각 폴더별 데이터 양 체크
    all_image = glob.glob(os.path.join(image_path,'*','*.jpg'))
    # print("all_images >>",len(all_image))    

    # 오렌지
    image_orange = glob.glob(os.path.join(image_path,'orange','*.jpg'))
    # print("orange images >>",len(image_orange))
    # print("orange",image_orange )
    # 자몽
    image_grapefruit = glob.glob(os.path.join(image_path,'grapefruit','*.jpg'))
    # print("grapefruit images >>",len(image_grapefruit))
    # 레드향
    image_kanpei = glob.glob(os.path.join(image_path,'kanpei','*.jpg'))
    # print("kanpei images >>",len(image_kanpei))
    # 한라봉
    image_dekopon = glob.glob(os.path.join(image_path,'dekopon','*.jpg'))
    # print("dekopon images >>",len(image_dekopon))
    

    
if __name__ =="__main__":
    image_file_check(image_path)