#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import cv2
import os
import sys
sys.path.append('../')
import random
from matplotlib import pyplot as plt
from BCNN.visualization.dataset import *

def data_analysis():

    root_path = '/home/speciallan/Documents/python/data/cloth/'
    origin_path = root_path + 'origin/'
    splitted_path = root_path + 'splitted/'
    test_path = root_path + 'test/mytest/'
    hist_path = root_path + 'test/mytest_hist/'

    filelist = os.listdir(test_path)
    filelist.sort()

    for c in filelist:
        class_dir_path = test_path + c

        # 打乱
        class_img_list = os.listdir(class_dir_path)
        class_img_list.sort()

        for file in class_img_list:
            img_path = os.path.join(test_path, c, file)
            save_path = img_path.replace('mytest', 'mytest_hist')
            show_gray(img_path, save_path)
            print('show {} done'.format(save_path))


if __name__ == '__main__':

    data_analysis()