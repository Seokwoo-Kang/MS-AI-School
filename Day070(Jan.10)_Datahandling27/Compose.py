import glob
import os
from PIL import Image


def image_synthesis(label):
    img_path=f"./del_back/{label}"
    img_list = glob.glob(os.path.join(img_path,"*.png"))
    back_list = glob.glob(os.path.join("./background/","*.jpeg"))    
    cnt = 0
    for i in img_list:
        for j in back_list:
            
            my_image = Image.open(j)
            my_image = my_image.resize((224, 224))
            watermark = Image.open(i)
            watermark = watermark.resize((224, 224))    # 배경제거된 상품 이미지 사이즈 결정
            x = my_image.size[0] - watermark.size[0]    # 새로운 배경에 넣을 좌표 설정 부분
            y = my_image.size[1] - watermark.size[1]
            my_image.paste(watermark, (x,y), watermark) # 배경에 이미지 합성
            my_image.save(f'./data/{label}/{label}_{cnt}.png')

            cnt += 1
img_path = "./del_back/"
label_list = ['Jinro', 'Choco']
for i in label_list:
    image_synthesis(i)


