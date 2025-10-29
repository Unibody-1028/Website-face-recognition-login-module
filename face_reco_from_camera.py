# time: 2024/7/2 0:55
# creater:guopengpeng


import dlib
import numpy as np
import cv2
import pandas as pd
import os
import time
from PIL import Image, ImageDraw, ImageFont

#Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

#Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        self.feature_known_list = []                #存放所有录入人脸特征的数组
        self.name_known_list = []                   #所有录入人脸的名字

        self.current_frame_face_cnt = 0             #当前摄像头中捕获到的人脸数
        self.current_frame_feature_list = []        #存储当前摄像头中捕获到的人脸特征
        self.current_frame_name_position_list = []  #存储当前摄像头中捕获到的所有人脸的名字坐标
        self.current_frame_name_list = []           #存储当前摄像头中捕获到的所有人脸的名字
        #FPS
        self.fps = 0
        self.frame_start_time = 0

    #从 "features_all.csv" 读取人脸特征
    def get_face_database(self):
        if os.path.exists("./data"):
            path_features_known_csv = "./data/features_total.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=0)
            print(len(csv_rd))

            csv_rd = csv_rd.iloc[:,1:]
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.feature_known_list.append(features_someone_arr)
                self.name_known_list.append("Person_"+str(i+1))
            print("Faces in Database：", len(self.feature_known_list))
            return 1
        else:
            print('##### Warning #####', '\n')
            print("'features_all.csv' not found!")
            print(
                "Please run 'get_faces_from_camera.py' and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'",
                '\n')
            print('##### End Warning #####')
            return 0

    #计算两个128D向量间的欧式距离
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    #更新FPS
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        #在人脸框下面写人脸名字
        font = ImageFont.truetype("simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_name_list[i], self.current_frame_name_position_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_name_position_list[i], text=self.current_frame_name_list[i], font=font)
            img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_with_name



    #处理获取的视频流，进行人脸识别
    def process(self, stream):
        #读取存放所有人脸特征的csv
        if self.get_face_database():
            while stream.isOpened():
                print(">>> Frame start")
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # 按下 q 键退出
                if kk == ord('q'):
                    break
                else:
                    self.draw_note(img_rd)
                    self.current_frame_feature_list = []
                    self.current_frame_face_cnt = 0
                    self.current_frame_name_position_list = []
                    self.current_frame_name_list = []

                    #检测到人脸
                    if len(faces) != 0:
                        #获取当前捕获到的图像的所有人脸的特征
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                        #遍历捕获到的图像中所有的人脸
                        for k in range(len(faces)):
                            print(">>>>>> For face", k+1, " in camera")
                            # 默认所有人不认识
                            self.current_frame_name_list.append("unknown")

                            #每个捕获人脸的名字坐标
                            self.current_frame_name_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            #对于某张人脸，遍历所有存储的人脸特征

                            current_frame_e_distance_list = []
                            for i in range(len(self.feature_known_list)):
                                #person_X不为空
                                if str(self.feature_known_list[i][0]) != '0.0':
                                    print("   >>> With person", str(i + 1), ", the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_feature_list[k],
                                                                                    self.feature_known_list[i])
                                    print(e_distance_tmp)
                                    current_frame_e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    current_frame_e_distance_list.append(999999999)
                            #寻找出最小的欧式距离
                            similar_person_num = current_frame_e_distance_list.index(min(current_frame_e_distance_list))
                            print("   >>> Minimum e distance with ", self.name_known_list[similar_person_num], ": ", min(current_frame_e_distance_list))
                            #如果欧氏距离小于0.4，则匹配成功
                            if min(current_frame_e_distance_list) < 0.4:
                                self.current_frame_name_list[k] = self.name_known_list[similar_person_num]
                                print("   >>> Face recognition result:  " + str(self.name_known_list[similar_person_num]))
                            else:
                                print("   >>> Face recognition result: Unknown person")

                            #矩形框
                            for kk, d in enumerate(faces):
                                #绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)

                        self.current_frame_face_cnt = len(faces)

                        #写名字
                        img_with_name = self.draw_name(img_rd)

                    else:
                        img_with_name = img_rd

                print(">>>>>> Faces in camera now:", self.current_frame_name_list)

                cv2.imshow("camera", img_with_name)

                #FPS
                self.update_fps()
                print(">>> Frame ends\n\n")

    #调用摄像头并进行 process
    def run(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("video.mp4")
        cap.set(3, 480)     # 640x480
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()