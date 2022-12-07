import cv2
import matplotlib.pyplot as plt

def imshow(src, windowName='show', close=False):
    cv2.imshow(windowName, src)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()

img_ori = cv2.imread('car.png')
rgb_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
# rgb_img = img_ori[:,:,::-1]
# img_ori[:,:,0]=0
# img_ori[:,:,1]=0
# img_ori[:,:,2]=0
# imshow(img_ori)

# cv2.imshow('origin', img_ori)
# cv2.imshow('origin2', img_ori)
# key = cv2.waitKey(0)
#
# if key == ord('q'):
#     print('key is q')


imshow(img_ori, 'show')


height, width, channel = img_ori.shape
print(height, width, channel)

# Convert Image to grayscale
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(img_ori)
plt.subplot(1,2,2)
plt.imshow(img_gray)

