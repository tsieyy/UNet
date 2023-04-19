import copy
import math

import cv2
import matplotlib.pyplot as plt

from _04Predict import *
'''
用opencv进行视频的逐帧获取，逐帧处理图片，用opencv对识别到的车道线求中点坐标，输出在视频图像上
'''
video = cv2.VideoCapture("test.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
index = 1
videoWriter = cv2.VideoWriter('trans.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)


def find_center(img):
    # print(groundtruth.shape)(768, 1024)
    h1, w1 = img.shape
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    contours, cnt = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩,字典形式
        if M["m00"] == 0:
            center_x = int(M["m10"])
            center_y = int(M["m01"])
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        # print(f'({center_x}, {center_y})')
        return center_x, center_y




lr = LaneLineRecognition()

success, frame = video.read()
X, Y = 0, 0
while success:
    frameCopy = copy.copy(frame)
    frameCopy = cv2.resize(frameCopy, (640, 480))

    frameCopy = lr.recognition(frameCopy)
    rol = frameCopy[120:520, 300:480]
    if find_center(rol) is not None:
        y, x = find_center(rol)
        # print(f'({x + 120}, {y + 300})')
        y = int((y + 300) / 640 * 1920)
        x = int((x + 120) / 480 * 1080)
        print(f'({y}, {x})')
    else:
        x, y = X, Y
        print(f'({y}, {x})')
    # print(frameCopy, frameCopy.shape)

    cv2.putText(frame, 'fps: ' + str(fps), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    cv2.putText(frame, 'count: ' + str(frameCount), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    cv2.putText(frame, 'frame: ' + str(index), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    cv2.putText(frame, 'time: ' + str(round(index / 24.0, 2)) + "s", (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    cv2.putText(frame, f'coordinate: ({y}, {x})', (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
    cv2.imshow("video", frame)
    cv2.waitKey(1000 // int(fps))
    videoWriter.write(frame)

    success, frame = video.read()
    index+=1
    X = x
    Y = y
video.release()