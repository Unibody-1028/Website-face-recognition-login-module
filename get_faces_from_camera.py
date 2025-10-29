# time: 2024/6/30 21:08
# creater:guopengpeng

import dlib
import numpy as np
import cv2
import os
import shutil
import time

#Dlib正向人脸检测器
detector = dlib.get_frontal_face_detector()

class Face_Register:
    def __init__(self):
        self.path_photos_from_camera = 'data/data_faces_from_camera/'
        self.font = cv2.FONT_ITALIC

        self.existing_faces_cnt = 0 #已录入不同人的人脸信息计数器
        self.ss_cnt = 0 #录入人脸时的图片数量计数器
        self.current_frame_faces_cnt = 0 #视频帧中识别出的人脸个数

        self.save_flag = 1 #用于控制是否保存图像的flag
        self.press_n_flag = 0 #用于检查先按'n'还是's'

        #FPS显示
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0

    #新建保存人脸图像文件和数据csv文件夹
    def pre_work_mkdir(self):

        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.makedirs(self.path_photos_from_camera)

    #删除之前存储人脸数据的文件夹
    def pre_work_del_old_face_folders(self):
        folders_rd = os.listdir(self.path_photos_from_camera) #列出指定路径下所有的文件和文件夹名称
        for i in range(len(folders_rd)):       #获取指定路径下文件的长度
            shutil.rmtree(self.path_photos_from_camera+'/'+folders_rd[i]) #递归删除目录及内容
        if os.path.isfile("../SDAUAi/data/features_all.csv"):  #如果该路径下有csv文件，则将其移除
            os.remove("../SDAUAi/data/features_all.csv")

    #如果之前有录入的人脸，在之前person_x的序号按照person_x+1开始录入
    def check_existing_face_cnt(self):
        if os.listdir("data/data_faces_from_camera"):
            #获取已经录入的最后一个人的人脸序号
            person_list = os.listdir("data/data_faces_from_camera")#列出文件夹下所有子文件夹的名字，并返回一个list
            person_num_list = []
            for person in person_list: #遍历所有子文件夹，并取出最大编号
                person_num_list.append(int(person.split('_')[-1]))
            self.existing_faces_cnt = max(person_num_list)
        #如果第一次存储或者没有之前录入的人脸，按照person_1开始录入
        else:
            self.existing_faces_cnt = 0

    #计算并更新每秒帧数
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now


    #在cv2window上添加说明文字
    def draw_note(self,img_rd):

        cv2.putText(img_rd, "Face_Recognize", (200,25), self.font, 1, (255,255,255),
                    1, cv2.LINE_AA)
        cv2.putText(img_rd,"FPS: "+str(self.fps.__round__(2)),(0,20),self.font, 0.8,(0,255,0),
                    1, cv2.LINE_AA)
        cv2.putText(img_rd,"Face_number: "+str(self.current_frame_faces_cnt),(0,40),self.font,0.8,
                    (0,255,0),1,cv2.LINE_AA)
        cv2.putText(img_rd, "N:Create face folder",(0,425),self.font, 0.8,(255,255,255),
                    1,cv2.LINE_AA)
        cv2.putText(img_rd, "S:Save current face", (0,450), self.font, 0.8, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.putText(img_rd,"Q:Quit",(0,475),self.font,0.8,(255,255,255),1,cv2.LINE_AA)


    #获取人脸
    def process(self,stream):
        #新建储存人脸图像文件目录
        self.pre_work_mkdir()

        #删除人脸图像文件目录中已有的文件
        if os.path.isdir(self.path_photos_from_camera):
            self.pre_work_del_old_face_folders()

        #检查人脸图像文件夹中是否已有人脸文件，以确定新文件夹序号
            self.check_existing_face_cnt()

        while stream.isOpened(): #视频流打开时，尝试读取帧
            flag,img_rd = stream.read() #从视频流中读取一帧,img_rd为读取的帧图像
            kk = cv2.waitKey(1) #等待键盘输入
            faces = detector(img_rd,0) #使用detector对象在img_rd上检测人脸

            #按下'n'新建存储人脸的文件夹
            if kk == ord('n'):
                self.existing_faces_cnt += 1
                #新建文件夹存储人脸图像
                current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
                os.makedirs(current_face_dir)
                print('\n')
                print("新的人脸文件夹：",current_face_dir)

                self.ss_cnt = 0 #将负责文件夹内计数的人脸图片计数器重置
                self.press_n_flag = 1 #已经按下'n'键

            #人脸检测部分
            if len(faces) != 0: #检测到人脸时

                for k,d in enumerate(faces): #k为索引，d为人脸的位置（左上和右下的坐标）
                    #计算矩形框大小
                    #左上角坐标为(0,0)
                    height = (d.bottom() - d.top()) #检测框的高度
                    width = (d.right() - d.left())  #检测框的宽度
                    hh = int(height/2)
                    ww = int(width/2)


                    #判断人脸检测框是否超出640x480的范围
                    """
                    矩形框的右边界加上其宽度的一半（`d.right()+ww`）是否超出了图像的右边界（640）。  
                    矩形框的下边界加上其高度的一半（`d.bottom()+hh`）是否超出了图像的下边界（480）。  
                    矩形框的左边界减去其宽度的一半（`d.left()-ww`）是否超出了图像的左边界（0）。  
                    矩形框的上边界减去其高度的一半（`d.top()-hh`）是否超出了图像的上边界（0）。
                    """
                    if(d.right()+int(0.5*ww) > 640) or (d.bottom()+int(0.5*hh) > 480) or (d.left()-int(0.5*ww) < 0) or (d.top()-int(0.5*hh) < 0):
                        cv2.putText(img_rd,"Out of range",(20,300),self.font,0.8,(0,0,255),
                                    1,cv2.LINE_AA)
                        color_rectangle = (0,0,255)
                        save_flag = 0
                        if kk ==ord('s'):
                            print("请调整位置")
                    else:
                        color_rectangle = (255,255,255)
                        save_flag = 1
                        #如果检测框未超出边界，则绘制检测框
                        cv2.rectangle(img_rd,
                                      tuple([d.left()-int(0.5*ww),d.top()-int(0.5*hh)]), #左上角
                                      tuple([d.right()+int(0.5*ww),d.bottom()+int(0.5*hh)]),#右下角
                                      color_rectangle,
                                      2
                                      )

                    #根据人脸大小生成空的图像
                    img_blank = np.zeros((int(height+0.5*hh),int(width+0.5*ww),3),np.uint8)

                    if save_flag:
                        #按下's'保存摄像头中的人脸到本地
                        if kk == ord('s'):
                            #检查有没有新建文件夹
                            if self.press_n_flag:
                                self.ss_cnt += 1
                                for i in range(height+int(0.5*hh)):
                                    for j in range(width+int(0.5*ww)):
                                        img_blank[i][j] = img_rd[d.top()-int(0.5*hh)+i+5][d.left()-int(0.5*ww)+j+5]

                                cv2.imwrite(current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg",img_blank)
                                print("写入本地",str(current_face_dir)+"/img_face_" + str(self.ss_cnt)+".jpg")
                            else:
                                print("请先建立文件夹")

            self.current_frame_faces_cnt = len(faces)

            #
            self.draw_note(img_rd)

            #按下q键退出
            if kk == ord('q'):
                break

            #更新FPS
            self.update_fps()

            cv2.namedWindow("camera",1)
            cv2.imshow("camera",img_rd)

    def run(self):
        cap = cv2.VideoCapture(0) #创建VideoCapture对象
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()






























