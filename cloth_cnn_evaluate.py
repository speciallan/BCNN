#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
sys.path.append('../')
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
from data_loader import build_generator, generator
from BCNN.models import model_zoo

img_width, img_height = 224,224
img_width, img_height = 224,40

# model = model_zoo.resnet50(shape=(img_height, img_width, 3))
# model = model_zoo.cbam(shape=(img_height, img_width, 3))
model = model_zoo.resnet50_se(shape=(img_height, img_width, 3))

# for layer in model2.layers[:]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
#     layer.trainable = True

weights_path = './model/cloth_resnet50_se.h5'
model.load_weights(weights_path, by_name=True)
print('Model loaded.')

test_data_dir = '../../data/cloth/splitted/test'
# abc 0.867 0.705 0.893
# 123 0.896 0.834 0.865
test_data_dir = '../../data/cloth/test/test_b'
# total = 32130
# total = 3257
total = 200
# total = 56
# batch_size = 512
batch_size = 16
classes = ['01', '02', '99']

# -------------------------------------------------------------------

test_generator = build_generator(
    train_dir=test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

test_generator = test_generator[0]
# t = next(test_generator)
# print(t[0].shape, t[1].shape)
# exit()

sum = 0
tp = 0
wrong = [0, 0, 0]
total_iters = total // batch_size + 1
for i in range(total_iters):
    print('正在评估第 {}/{} 个循环'.format(i+1, total_iters))
    data = next(test_generator)
    test_imgs = data[0]
    true_labels = data[1]

    # result = model.predict(test_imgs, batch_size=batch_size, verbose=0)
    result = model.predict_on_batch(test_imgs)

    # 输出
    true_labels = np.argmax(true_labels, axis=-1)
    pred_labels = np.argmax(result, axis=-1)

    sum += len(true_labels)

    # 评估
    for k,v in enumerate(true_labels):
        if true_labels[k] == pred_labels[k]:
            tp += 1
        else:
            print('true:{}, wrong:{}, prob:{}'.format(true_labels[k], pred_labels[k], result[k]))
            wrong[true_labels[k]] += 1

print('total: {}, acc: {:.3f}'.format(sum, tp*1.0/sum))
print('wrong: {}'.format(wrong))

exit()
# -------------------------------------------------------------------

test_generator = generator(test_data_dir, classes=classes, batch_size=batch_size, target_size=(img_width, img_height))

# t = next(test_generator)
# print(t[0].shape)
# exit()

tp = 0
wrong = [0, 0, 0]
total_iters = total // batch_size + 1
for i in range(total_iters):
    print('正在评估第 {}/{} 个循环'.format(i, total_iters))
    data = next(test_generator)
    test_imgs = data[0]
    true_labels = data[1]
    filepaths = data[2]

    result = model.predict_on_batch(test_imgs)

    # 输出
    pred_labels = np.argmax(result, axis=-1)

    # 评估
    for k,v in enumerate(true_labels):
        if true_labels[k] == pred_labels[k]:
            tp += 1
        else:
            print('true:{}, pred:{}, prob:{}, filepath:{}'.format(true_labels[k], pred_labels[k], result[k], filepaths[k]))
            wrong[true_labels[k]] += 1

print('total: {}, acc: {:.3f}'.format(total, tp*1.0/total))
print('wrong: {}'.format(wrong))

exit()








test_generator = generator(test_data_dir, classes=classes, batch_size=batch_size, target_size=(img_height, img_width))

all_test_labels = []
all_pred_labels = []
next_flag = True

i = 0
tp = 0
while next_flag:

    if i > total:
        break

    imgs, labels, filepaths, next_flag = next(test_generator)
    result = model2.predict_on_batch(imgs)

    pred_labels = np.argmax(result, axis=-1)

    all_test_labels.extend(labels)
    all_pred_labels.extend(pred_labels)

    # print(imgs.shape)
    # print(all_test_labels, all_pred_labels)
    i += len(imgs)
    print('正在评估第 {} 条数据'.format(i, total))

    for k,v in enumerate(labels):
        if labels[k] == pred_labels[k]:
            tp += 1
        else:
            print('true:{}, wrong:{}, prob:{}, path:{}'.format(labels[k], pred_labels[k], result[k], filepaths[k]))

print('total: {}, acc: {} / {} = {:.3f}'.format(i, tp, i, tp*1.0/i))


