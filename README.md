# Website-face-recognition-login-module

人脸识别的流程：
	从摄像头采集人脸图像并存储，通过识别出人脸的68个特征点，计算提取人脸的128D特征，存储到csv文件，然后与摄像头中实时采集的人脸所提取出的特征进行对比，计算两者的欧氏距离，以判断是否是同一个人

识别模型：基于Dlib的ResNet预训练模型（dlib_face_recognition_resnet_model_v1.dat）



