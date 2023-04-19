# 去畸变处理,若图片是由小车的摄像头进行拍摄的，那需要进行这一步

import glob, cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)


ImgPaths = glob.glob(sys.path[0][:sys.path[0].rindex('\\')] + "\\dataset_raw\\OriImage\\*")

if not ImgPaths:
    print('wrong path')
    exit(-1)

H = np.array([[-0.61869793, -2.24344654, 672.50410257],
              [0.00583877, 0.05218149, -226.93620917],
              [-0.00011433, -0.00451613, 1.]])

inv_H = np.array([[-1.79141884e+00, -1.46171084e+00,  8.73021405e+02],
                  [3.70310301e-02, -9.97859211e-01, -2.51353906e+02],
                  [-3.75759704e-05, -4.67357932e-03, -3.53343796e-02]])
# 相机参数
Dist = np.array([-0.26538, 0.08153, -0.00109, -0.00233, 0.00000], dtype=np.float32)
K = np.array([[331.71415, 0, 321.54719],
              [0, 331.80738, 201.23948],
              [0, 0, 1]], dtype=np.float32)



for ImgPath in ImgPaths:
    print(ImgPath)
    Img = cv2.imread(ImgPath)
    UndistImg = cv2.undistort(Img, K, Dist)
    #  如果需要透视变换的话就取消注释这个
    # WarpedImg = cv2.warpPerspective(UndistImg, H, (1000, 1000))
    SavePath = ImgPath.replace('OriImage', 'UndistImg')
    os.makedirs(os.path.dirname(SavePath), exist_ok=True)
    cv2.imwrite(SavePath, UndistImg)
