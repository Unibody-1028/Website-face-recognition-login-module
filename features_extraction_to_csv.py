# time: 2024/7/1 2:16
# creater:guopengpeng


import os
import dlib
import csv
import numpy as np
import logging
import cv2

#要读取的图像的路径
images_path = "data/data_faces_from_camera/"
#Dlib人脸检测器
detector = dlib.get_frontal_face_detector()
#人脸特征点检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#人脸识别模型，提取128D的特征矢量
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
#print(face_rec_model)

#返回单张图像的128D特征
def retrun_128d_features(path_imgs:str)->dlib.vector:
    """
    本函数负责提取单张面部图像的128D特征
    :param path_imgs:人脸图像文件路径
    :return:人脸128D特征向量
    """
    img_rd = cv2.imread(path_imgs)
    faces = detector(img_rd,1)
    #记录日志信息
    logging.info("%-40s %-20s","检测到人脸图像",path_imgs)

    if len(faces) != 0:
        shape = predictor(img_rd,faces[0])
        face_descriptor = face_rec_model.compute_face_descriptor(img_rd,shape)
    else:

        face_descriptor = 0
        logging.warning("no faces")

    #print(face_descriptor)

    return face_descriptor

#retrun_128d_features('data/data_faces_from_camera/person_1/img_face_1.jpg')

#返回某个人的128D特征均值

def return_features_mean_person(path_face_personX):
    """

    :param path_face_personX:
    :return:
    """
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for i in range(len(photos_list)):

            features_128d = retrun_128d_features(path_face_personX+"/"+photos_list[i])
            if features_128d == 0: #没有检测出人脸就跳过这张图片
                i += 1
            else:
                features_list_personX.append(features_128d)
    else:
        logging.warning("文件夹内图像文件为空：",path_face_personX)

    #计算128D特征的均值
    if features_list_personX:
        features_mean_personX = np.array(features_list_personX,dtype=object).mean(axis=0)#列表嵌套
    else:
        features_mean_personX = np.zeros(128,dtype=int,order='C')
    return features_mean_personX



def main():
    logging.basicConfig(level=logging.INFO)
    #获取已经录入的最有一个人脸序号
    person_list = os.listdir('data/data_faces_from_camera')
    #print(person_list)
    person_list.sort()

    with open('./data/features_total.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # 写入列名，第一列是姓名，后面是128个特征列
        header = ['person_name'] + ['feature_{}'.format(i) for i in range(128)]
        writer.writerow(header)

    with open('./data/features_total.csv', "a", newline="") as csvfile:  # 使用 "a" 模式追加数据
        writer = csv.writer(csvfile)
        for person in person_list:
            #print(person)
            #print(images_path)
            logging.info(f"{images_path}{person}")
            features_mean_personX = return_features_mean_person(images_path+person)

            if len(person.split('_',2)) == 2:
                person_name = person
            else:
                person_name = person.split('_',2)[-1]
            features_mean_personX = np.insert(features_mean_personX,0,person_name,axis=0)

            writer.writerow(features_mean_personX)
            logging.info('\n')
        logging.info("所有人脸128D数据已存储")


if __name__ == '__main__':
    main()
