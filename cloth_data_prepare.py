#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
import cv2
import os
import random

def data_prepare():

    root_path = '/home/speciallan/Documents/python/data/cloth/'
    origin_path = root_path + 'origin/'
    splitted_path = root_path + 'splitted/'

    filelist = os.listdir(origin_path)
    filelist.sort()

    train_test_split = 0.5
    train_valid_split = 0.8

    for c in filelist:
        class_dir_path = origin_path + c

        # 打乱
        class_img_list = os.listdir(class_dir_path)
        random.shuffle(class_img_list)

        # 划分
        total = len(class_img_list)
        trainval_list = class_img_list[:int(total*train_test_split)]
        total_trainval = len(trainval_list)
        train_list = trainval_list[:int(total_trainval*train_valid_split)]
        valid_list = trainval_list[int(total_trainval*train_valid_split):]
        test_list = class_img_list[int(total*train_test_split):]

        # 7195 1799 8994
        # print(len(train_list), len(valid_list), len(test_list))
        # exit()

        # 复制到新文件夹
        for c2 in ['train', 'valid', 'test']:

            if not os.path.exists(os.path.join(splitted_path, c2)):
                os.makedirs(os.path.join(splitted_path, c2))

            for file in eval(c2+'_list'):
                from_path = os.path.join(origin_path, c, file)
                to_path = os.path.join(splitted_path, c2, c, file)

                if not os.path.exists(os.path.join(splitted_path, c2, c)):
                    os.makedirs(os.path.join(splitted_path, c2, c))

                # img = cv2.imread(from_path)
                # img = cv2.resize(img, (224,224))
                # cv2.imwrite(to_path, img)
                os.system('cp {} {}'.format(from_path, to_path))

                print(to_path)

    print('数据集生成完毕')


if __name__ == '__main__':

    data_prepare()