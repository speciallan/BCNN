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
from data_loader import build_generator, generator

img_width, img_height = 224,224

#model = VGG16(include_top=False, weights='imagenet')

input_tensor = Input(shape=(img_width, img_height, 3)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

num_classes = 3
x = model.output
x = GlobalAvgPool2D(name='avg_pool')(x)
x = Dense(num_classes, activation='softmax', name='fc')(x)

model2 = Model(inputs=model.input, outputs=x)

weights_path = './model/cloth_resnet50.h5'
model2.load_weights(weights_path, by_name=True)
print('Model loaded.')

test_data_dir = '../../data/cloth/splitted/test'
# abc 0.867 0.705 0.893
# 123 0.896 0.834 0.865
test_data_dir = '../../data/cloth/test/1'
total = 32130
# total = 1000
# total = 1000
batch_size = 256
classes = ['01', '02', '99']

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
    # result = model2.predict(imgs, batch_size=batch_size, verbose=0)
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
            # print(type(labels[k]), labels[k], type(pred_labels[k]), pred_labels[k])
            continue
            print('true:{}, wrong:{}, prob:{}, path:{}'.format(labels[k], pred_labels[k], result[k], filepaths[k]))

print('total: {}, acc: {} / {} = {:.3f}'.format(i, tp, i, tp*1.0/i))

