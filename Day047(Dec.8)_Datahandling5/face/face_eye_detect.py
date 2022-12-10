import cv2
import numpy as np

# Creating face_cascade and eye_cascade objects
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

face_img = cv2.imread('face01.png')

# Converting the image into grayscale
face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# Creating variable faces
# detectMultiScale(gray image, double scaleFactor, int minNeighbors)
faces = face_cascade.detectMultiScale(face_gray, 1.1, 4)

# Defining and drawing the rectangle around the face
for (x,y,w,h) in faces:
    cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 3)

# Creating two regions of interest
roi_img = face_img[y:(y + h), x:(x+w)]
roi_gray = face_gray[y:(y + h), x:(x+w)]

# Creating variable eyes
eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
index=0
for (ex,ey,ew,eh) in eyes:
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_2 = (ex, ey, ew, eh)

# # Drawing rectangle around the eyes
    cv2.rectangle(roi_img, (ex, ey), (ex+ew, ey+eh), (0,0,255), 3)
    index += 1

if eye_1[0] < eye_2[0]:
    left_eye = eye_1
    right_eye = eye_2
else:
    left_eye = eye_2
    right_eye = eye_1

# Calculating coordinates of a central points of the rectangles
left_eye_center = (int(left_eye[0] + (left_eye[2]/2)),
                   int(left_eye[1] + (left_eye[3]/2)))
left_eye_x = left_eye_center[0]
left_eye_y = left_eye_center[1]

right_eye_center = (int(right_eye[0] + (right_eye[2]/2)),
                    int(right_eye[1] + (right_eye[3]/2)))
right_eye_x = right_eye_center[0]
right_eye_y = right_eye_center[1]

cv2.circle(roi_img, left_eye_center, 5, (255, 0, 0), -1)
cv2.circle(roi_img, right_eye_center, 5, (255, 0, 0), -1)
cv2.line(roi_img, right_eye_center, left_eye_center, (0, 200, 200), 3)

if left_eye_y > right_eye_y:
    A = (right_eye_x, left_eye_y)
    # Integer -1 indicates that the image will rotate in the clockwise direction
    direction = -1
else:
    A = (left_eye_x, right_eye_y)
    # Integer 1 indicates that the image will rotate in the counter clockwise direction
    direction = 1

cv2.circle(roi_img, A, 5, (255,0,0), -1)

cv2.line(roi_img, right_eye_center, left_eye_center, (0, 200, 200), 3)
cv2.line(roi_img, left_eye_center, A, (0, 200, 200), 3)
cv2.line(roi_img, right_eye_center, A, (0, 200, 200), 3)

# np.arctan 함수는 라디안 단위로 각도를 반환 한다는 점에 유의결과를 각도로 변환 하려면
# 각도(세타)에 180을 곱한 다음 원주율로 나누어야 합니다.
delta_x = right_eye_x - left_eye_x
delta_y = right_eye_y - left_eye_y
angle = np.arctan(delta_y/delta_x)
angle = (angle * 180) /np.pi

# 이미지를 각도 만큼 회전
# Width and height of the image
h, w = face_img.shape[:2]

# Calculating a center point of the image
# Integer division "//" ensures that we receive whole number
center = (w//2, h//2)
# Defining a matrix M and calling
# cv2.getRotationMatrix2D method
M = cv2.getRotationMatrix2D(center, (angle), 1.0)
# Applying the rotation to our image using
# cv2.warpAffine method
rotated = cv2.warpAffine(face_img, M, (w,h))

dist_1 = np.sqrt((delta_x*delta_x)+(delta_y*delta_y))

# dist_2를 계산하기 위한 과정
# 회전시킨 이미지(rotated)에서 양쪽 눈의 좌표를 얻고 과정 반복
# Creating variable rotated eyes
eyes_rot = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
index=0
for (ex,ey,ew,eh) in eyes_rot:
    if index == 0:
        eye_rot1 = (ex, ey, ew, eh)
    elif index == 1:
        eye_rot2 = (ex, ey, ew, eh)
    index += 1

if eye_1[0] < eye_2[0]:
    left_eye_rot = eye_rot1
    right_eye_rot = eye_rot2
else:
    left_eye_rot = eye_rot2
    right_eye_rot = eye_rot1

left_eye_rot_center = (int(left_eye_rot[0] + (left_eye_rot[2]/2)),
                   int(left_eye_rot[1] + (left_eye_rot[3]/2)))
left_eye_rot_x = left_eye_rot_center[0]
left_eye_rot_y = left_eye_rot_center[1]

right_eye_rot_center = (int(right_eye_rot[0] + (right_eye_rot[2]/2)),
                    int(right_eye_rot[1] + (right_eye_rot[3]/2)))
right_eye_rot_x = right_eye_rot_center[0]
right_eye_rot_y = right_eye_rot_center[1]

delta_x_1 = right_eye_rot_x - left_eye_rot_x
delta_y_1 = right_eye_rot_y - left_eye_rot_y

dist_2 = np.sqrt((delta_x_1*delta_x_1)+(delta_y_1*delta_y_1))

# Calculate the ratio
ratio = dist_1 / dist_2

# Defining the width and height
h=476
w=488

# Defining aspect ratio of a resized image
dim =(int(w*ratio), int(h*ratio))

resized = cv2.resize(rotated, dim)

cv2.imshow("face_rotated", rotated)
cv2.imshow("face_resized", resized)
cv2.waitKey(0)

# cv2.imwrite('./rotated_01.png', rotated)
# cv2.imwrite('./resized_01.png', resized)