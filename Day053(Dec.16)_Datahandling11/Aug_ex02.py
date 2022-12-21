# 이미지 사이즈 -> 바운딩 위치
import cv2
import numpy as np

def cvTest():
    image = cv2.imread('./Agumentation/01.jpg')
    y_ = image.shape[0]
    x_ = image.shape[1]

    target_size = 600
    x_scale = target_size / x_
    y_scale = target_size / y_
    # print("x_scale >>", x_scale, "y_scale >>", y_scale)
    # x_scale >> 0.4 y_scale >> 0.5333333333333333

    img = cv2.resize(image, (target_size, target_size))
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    bboxes = [[3.96, 183.38, 200.88, 214.03],[468.94, 92.01, 171.06, 248.45]]
    for boxs in bboxes:
        x_min, y_min, w, h = boxs

        # xywh to x1y1x2y2
        x_min, x_max, y_min, y_max = int(x_min), int(x_min+w), int(y_min), int(y_min+h)

        x1 = int(np.round(x_min*x_scale))
        y1 = int(np.round(y_min*y_scale))
        x2 = int(np.round(x_max*x_scale))
        y2 = int(np.round(y_max*y_scale))

        cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 1)
    cv2.imshow("test", img)
    cv2.waitKey(0)    


if __name__ =='__main__':
    cvTest()