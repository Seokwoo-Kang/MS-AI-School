import cv2

cap = cv2.VideoCapture(0)
cnt = 0
if cap.isOpened():
    while True:
        ret, fram = cap.read()
        if ret:
            cv2.imshow("camera",fram) #프레임 이미지 표시
            if cv2.waitKey(1) != -1:
                cv2.imwrite("./0106/image/photo" + "{0:04d}".format(cnt) + ".jpg",fram)
                cnt += 1
        else:
            print("no fram")
            break
else:
    print("can't open camera")
cap.release()
cv2.destroyAllWindows()