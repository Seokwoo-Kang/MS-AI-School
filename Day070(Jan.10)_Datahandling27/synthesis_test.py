import cv2
import numpy as np
import glob
import os
from PIL import Image

large_img = cv2.imread('./del_back/000.png')
watermakr = cv2.imread('./utils/background/3.jpeg')

print("large_image size >> ", large_img.shape)
print("watermakr image size >> ", watermakr.shape)

img1 = cv2.resize(large_img, (800, 600))
img2 = cv2.resize(watermakr, (800, 600))

print("img1 reize >>", img1.shape)
print("img2 reize >>", img2.shape)

"""
large_image size >>  (683, 1024, 3)
watermakr image size >>  (480, 640, 3)
img1 reize >> (600, 800, 3)
img2 reize >> (600, 800, 3)
"""
# 혼합 진행

# # 베이스 5:5
blended = cv2.addWeighted()
img1, 0.5, img2, 0.5, 0

# 9:1
# blended = cv2.addWeighted(img1, 9, img2, 1, 0)

# 1로 설정
# blended = cv2.addWeighted(img1, 1, img2, 1, 0)
cv2.imshow("image show", blended)
cv2.waitKey(0)



# class image_synthesis(file_path):
#     def __init__(self, file_path):
#         pass

#     def __len__(self):
#         return len(self.)

def image_synthesis(file_path):
    img_list = []
    img_list = glob.glob(os.path.join(file_path,"*.png"))
    back_list = []
    back_list = glob.glob(os.path.join("./utils/background/","*.jpeg"))    
    
    for i in img_list:
        for j in back_list:

            my_image = Image.open(j)  
            watermark = Image.open(i)
            watermark = watermark.resize((160, 160)) # 230px 60px 로 워터마크 사진 크기 조절
            x = my_image.size[0] - watermark.size[0]
            y = my_image.size[1] - watermark.size[1]
            my_image.paste(watermark, (x,y), watermark)
            my_image.show()

            img = cv2.imread(i) #사과파일 읽기
            img1 = cv2.resize(img, (800, 600))
            # print(img)
            back = cv2.imread(j) #로고파일 읽기
            back1 = cv2.resize(back, (800, 600))
            blended = cv2.addWeighted(img1, 0.5, back1, 0.5, 0)
            
            dst = cv2.bitwise_or(img1, back1) #src1_bg와 src2_fg를 합성

            cv2.imshow("image show", blended)
            cv2.waitKey(0)

            # # print(back)           
            # rows, cols, channels = back.shape #로고파일 픽셀값 저장
            # roi = img[:rows,:cols] #로고파일 필셀값을 관심영역(ROI)으로 저장함.
            
            # gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY) #로고파일의 색상을 그레이로 변경
            # ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY) #배경은 흰색으로, 그림을 검정색으로 변경
            # mask_inv = cv2.bitwise_not(mask)
            # # cv2.imshow('mask',mask) #배경 흰색, 로고 검정
            # # cv2.imshow('mask_inv',mask_inv) # 배경 검정, 로고 흰색
            
            # img_bg = cv2.bitwise_and(roi,roi,mask=mask) #배경에서만 연산 = src1 배경 복사
            # # cv2.imshow('img_bg',img_bg)
            
            # back_fg = cv2.bitwise_and(back,back, mask = mask_inv) #로고에서만 연산
            # # cv2.imshow('back_fg',back_fg)
            
            # dst = cv2.bitwise_or(img_bg, back_fg) #src1_bg와 src2_fg를 합성
            # # cv2.imshow('dst',dst)
            
            # img[:rows,:cols] = dst #src1에 dst값 합성
            
            # cv2.imshow('result',img)
            # cv2.waitKeyEx()
            # cv2.destroyAllWindows()



# image_synthesis("./del_back")
# exit()

