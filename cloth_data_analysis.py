#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import cv2
import os
import random
from matplotlib import pyplot as plt

def data_prepare():

    root_path = '/home/speciallan/Documents/python/data/cloth/'
    origin_path = root_path + 'origin/'
    splitted_path = root_path + 'splitted/'

    filelist = os.listdir(origin_path)
    filelist.sort()

    train_test_split = 0.5
    train_valid_split = 0.8

    w_arr, h_arr = [], []

    for c in filelist:
        class_dir_path = origin_path + c

        # 打乱
        class_img_list = os.listdir(class_dir_path)
        class_img_list.sort()

        for file in class_img_list:
            img = cv2.imread(os.path.join(origin_path, c, file))
            h, w, _ = img.shape
            w_arr.append(w)
            h_arr.append(h)
            # print(file, w, h)
            # exit()


        a = w_arr
        d_width = 50
        num_bins = int(max(a) - min(a)) // d_width
        plt.figure(figsize=(20, 8), dpi=80)
        # 频数分布直方图
        # 第二个参数是组数，
        # normed=True 这个代表某个时间代出现的频率
        plt.hist(a, num_bins)
        # 设置X轴刻度
        plt.xticks(range(min(a), max(a) + d_width, d_width))
        # 显示网格
        plt.grid()
        # plt.show()
        plt.savefig('./w.jpg')

        a = h_arr
        d_width = 50
        num_bins = int(max(a) - min(a)) // d_width
        plt.figure(figsize=(20, 8), dpi=80)
        # 频数分布直方图
        # 第二个参数是组数，
        # normed=True 这个代表某个时间代出现的频率
        plt.hist(a, num_bins)
        # 设置X轴刻度
        plt.xticks(range(min(a), max(a) + d_width, d_width))
        # 显示网格
        plt.grid()
        # plt.show()
        plt.savefig('./h.jpg')
        print('保存完毕')
        exit()



if __name__ == '__main__':

    data_prepare()