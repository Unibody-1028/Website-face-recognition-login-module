# time: 2024/6/30 19:47
# creater:guopengpeng
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

#读取图片
img_path = './face_recognize/1.jpg'
img = cv2.imread(img_path)
origin_img = img.copy()

#定义人脸检测器
detector = dlib.get_frontal_face_detector()

#定义人脸关键点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#检测得到的人脸
faces = detector(img,0)
#如果存在人脸
if(len(faces)):
    print("Found {} faces in this image.".format(len(faces)))
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x,p.y] for p in predictor(img,faces[i]).parts()])
        for point in landmarks:
            pos = (point[0,0],point[0,1])
            cv2.circle(img,pos,1,color=(0,255,255),thickness=3)
else:
    print("Not Found Face!")

cv2.namedWindow('Origin Face', cv2.WINDOW_FREERATIO)
cv2.namedWindow('Detected Face', cv2.WINDOW_FREERATIO)
cv2.imshow("Origin Face",origin_img)
# cv2.waitKey(0)
cv2.imshow("Detected Face",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

