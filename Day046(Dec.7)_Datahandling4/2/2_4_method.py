import cv2

# 확장, 침식 실험 예시 3
# Method
# Morphological gradient and Top hat operations : 형태학적 기울기 및 상단 모자 작업
"""
Gradient : dectect edge ( dilation - erosion)
Tophat : original - opening
Blackhat : closing - original
opening : dilation -> erosion
closing : erosion -> dilation
"""

image = cv2.imread('./Billiards.png', cv2.IMREAD_GRAYSCALE)

_, mask=cv2.threshold(image, 230, 255, cv2.THRESH_BINARY_INV)

op_idx = {
    'gradient' : cv2.MORPH_GRADIENT,
    'tophat' : cv2.MORPH_TOPHAT,
    'blackhat' : cv2.MORPH_BLACKHAT,
}

def onChange(k, op_name):
  if k == 0:
    cv2.imshow(op_name, mask)
    cv2.waitKey(0)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
  dst = cv2.morphologyEx(mask, op_idx[op_name], kernel)
  cv2.imshow(op_name,dst)

cv2.imshow('original', image)

cv2.imshow('gradient',mask)
cv2.imshow('tophat',mask)
cv2.imshow('blackhat',mask)

cv2.createTrackbar('k', 'gradient', 0, 300, lambda x: onChange(k=x, op_name='gradient'))
cv2.createTrackbar('k', 'tophat', 0, 300, lambda x: onChange(k=x, op_name='tophat'))
cv2.createTrackbar('k', 'blackhat', 0, 300, lambda x: onChange(k=x, op_name='blackhat'))