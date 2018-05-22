#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/5/17'
# 
"""
import cv2

# cap = cv2.VideoCapture(0)

c = cv2.imread('17_Ceremony_Ceremony_17_325.jpg')
height, width = c.shape[:2]
cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
cv2.resizeWindow('jpg', width, height)
cv2.imshow('jpg', c)
r = cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread(r'17_Ceremony_Ceremony_17_325.jpg')
cv2.imshow("Original", image)
# image[0:5, 0:5] = (0, 0, 255)
# cv2.imshow("Color1", image)
# image[0:5, 0:5] = (0, 255, 0)
# cv2.imshow("Color2", image)
# image[0:5, 0:5] = (255, 0, 0)
# cv2.imshow("Color3", image)

# 计算图像的中心点
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)

# 将图像平均分成四部分并显示
# tl = image[0:cY, 0:cX]
# tr = image[0:cY, cX:w]
# br = image[cY:h, cX:w]
# bl = image[cY:h, 0:cX]
# cv2.imshow("Top-Left Corner", tl)
# cv2.imshow("Top-Right Corner", tr)
# cv2.imshow("Bottom-Right Corner", br)
# cv2.imshow("Bottom-Left Corner", bl)

cv2.waitKey(0)