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


# img = cv2.imread(ImgPaths[0])
# im2 = cv2.resize(img,(1000,1000),)  # 为图片重新指定尺寸
# plt.imshow(im2)
# plt.show()

#去畸形化
# undisImg = cv2.undistort(im2, K, Dist)
# plt.imshow(undisImg)
# plt.show()

# warpImg = cv2.warpPerspective(im2, H, (1000, 1000))
# plt.imshow(warpImg)
# plt.show()


#求H的逆矩阵
# u, s, v = np.linalg.svd(H, full_matrices=False)#截断式矩阵分解
# inv_H = np.matmul(v.T * 1 / s, u.T)#求逆矩阵

# oriImg = cv2.warpPerspective(im2, inv_H, (640, 480))
# plt.imshow(oriImg)
# plt.show()

# plt.imshow(cv2.imread('OriImage/0000.jpg'))
# plt.show()



for ImgPath in ImgPaths:
    print(ImgPath)
    Img = cv2.imread(ImgPath)
    UndistImg = cv2.undistort(Img, K, Dist)

    WarpedImg = cv2.warpPerspective(UndistImg, H, (1000, 1000))
    SavePath = ImgPath.replace('OriImage', 'UndistImg')
    os.makedirs(os.path.dirname(SavePath), exist_ok=True)
    cv2.imwrite(SavePath, UndistImg)
