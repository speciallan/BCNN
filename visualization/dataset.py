#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_hist(img_path, save_path):
    plt.figure()
    img = cv2.imread(img_path, 0)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.savefig(save_path)
    # plt.show()

def show_rgb(img_path, save_path):
    plt.figure()
    img = cv2.imread(img_path, 1)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.savefig(save_path)
    # plt.show()

def show_gray(img_path, save_path):
    plt.figure()
    img = cv2.imread(img_path, 0)
    # 得到计算灰度直方图的值
    xy = xygray(img)

    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, xy, "r", linewidth=2, c='black')
    # 设置坐标轴的范围
    y_maxValue = np.max(xy)
    plt.axis([0, 255, 0, y_maxValue])
    # 设置坐标轴的标签
    plt.xlabel('gray Level')
    plt.ylabel("number of pixels")
    plt.savefig(save_path)
    # plt.show()

def xygray(img):
    # 得到高和宽
    rows, cols = img.shape
    # 存储灰度直方图
    xy = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            xy[img[r][c]] += 1
    # 返回一维ndarry
    return xy
