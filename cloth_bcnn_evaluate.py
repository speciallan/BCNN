#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAvgPool2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50, InceptionV3
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras import backend as K
from keras.models import Model
from data_loader import build_generator
from model_builder import build_bcnn

img_width, img_height = 224,224
img_width, img_height = 224,40
num_classes = 3

model = build_bcnn(
    all_trainable=True,
    size_height=img_height,
    size_width=img_width,
    no_class=num_classes,
)

weights_path = './model/bcnn.h5'
model.load_weights(weights_path, by_name=True)
print('Model loaded.')

test_data_dir = '../../data/cloth/splitted/test'
total = 32130
# total = 10000
batch_size = 256
classes = ['01', '02', '99']

test_generator = build_generator(
    train_dir=test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

test_generator = test_generator[0]
# t = next(test_generator)
# print(t[0].shape, t[1].shape)
# exit()

tp = 0
wrong = [0, 0, 0]
total_iters = total // batch_size
for i in range(total_iters):
    print('正在评估第 {}/{} 个循环'.format(i, total_iters))
    data = next(test_generator)
    test_imgs = data[0]
    test_labels = data[1]

    result = model.predict(test_imgs, batch_size=batch_size, verbose=0)

    # 输出
    test_labels = np.argmax(test_labels, axis=-1)
    pred_labels = np.argmax(result, axis=-1)

    # 评估
    for k,v in enumerate(test_labels):
        if test_labels[k] == pred_labels[k]:
            tp += 1
        else:
            print('true:{}, wrong:{}, prob:{}'.format(test_labels[k], pred_labels[k], result[k]))
            wrong[test_labels[k]] += 1

print('total: {}, acc: {:.3f}'.format(total, tp*1.0/total))
print('wrong: {}'.format(wrong))

