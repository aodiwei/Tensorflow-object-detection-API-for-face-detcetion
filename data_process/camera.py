#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/5/18'
# 
"""
import cv2

frame_num = 0
face_num = 0


def camera_detector():
    global frame_num
    cap = cv2.VideoCapture(0)  # 打开笔记本内置的摄像头

    while cap.isOpened():
        # time.sleep(1) #延迟x秒
        ret, frame = cap.read()  # 读取一帧，前一个返回值是是否成功，后一个返回值是图像本身
        #        out.write(frame) #把每帧图片一帧帧地写入video中
        show_image = face_detector(frame)  # show_image是返回的已标记出人脸的图片
        cv2.imshow("monitor", show_image)  # 在窗口显示一帧

        key = cv2.waitKey(40)
        if key == 27 or key == ord('q'):  # 如果按ESC或q键，退出
            break
        if key == ord('s'):  # 如果按s键，保存图片
            cv2.imwrite("frame_%s.png" % frame_num, frame)
            frame_num = frame_num + 1
            #    out.release()
    cap.release()
    cv2.destroyAllWindows()


# 检测出图片中的人脸，并用方框标记出来
def face_detector(image, cascade=None):
    global face_num  # 引用全局变量
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化图片 
    equalImage = cv2.equalizeHist(grayImage)  # 直方图均衡化
    # faces = cascade.detectMultiScale(equalImage, scaleFactor=1.3, minNeighbors=3)
    faces = [(449, 330, 122, 149)]
    for (x, y, w, h) in faces:
        # 裁剪出人脸，单独保存成图片，注意这里的横坐标与纵坐标不知为啥颠倒了
        # cv2.imwrite("face_%s.png" %(face_num), image[y:y+h,x:x+w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_num = face_num + 1
    return image


if __name__ == '__main__':
    camera_detector()
