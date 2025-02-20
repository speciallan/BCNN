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
from BCNN.visualization.grad_cam import draw

img_width, img_height = 224,40
img_width, img_height = 320,80

# model = model_zoo.resnet50(shape=(img_height, img_width, 3))
# model = model_zoo.cbam(shape=(img_height, img_width, 3))
# model = model_zoo.resnet50_se(shape=(img_height, img_width, 3))
# model = model_zoo.resnet101_se(shape=(img_height, img_width, 3))
model = model_zoo.xception(shape=(img_height, img_width, 3))

# for layer in model2.layers[:]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
#     layer.trainable = True

model.summary()
# exit()
model_name = 'xception'
last_layer_name = 'block14_sepconv2_act' # resnet
# model_name = 'resnet50'
# last_layer_name = 'activation_49' # resnet
# model_name = 'resnet50_se'
# last_layer_name = 'activation_50' # resnet50_se
# last_layer_name = 'activation_51' # resnet50_cbam

# model_name = 'cbam'
# last_layer_name = 'activation_73' # cbam
# model_name = 'resnet101_se'
# last_layer_name = 'activation_49' # se
# model_name = 'resnet101_se'
# last_layer_name = 'activation_103' # se

weights_path = './model/cloth_xception.h5'
weights_path = './model/cloth.h5'
# weights_path = './model/cloth_cbam.h5'
# weights_path = './model/cloth_resnet101_se.h5'
model.load_weights(weights_path, by_name=True)
print('Model loaded.')

test_data_dir = '../../data/cloth/test/mytest'
show_data_dir = '../../data/cloth/test/mytest_show'

total = 50
batch_size = 1
classes = ['01', '02', '99']

test_generator = generator(test_data_dir, classes=classes, batch_size=batch_size, target_size=(img_width, img_height))

# test_generator = build_generator(
#     train_dir=None,
#     valid_dir=test_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size)
#
# test_generator = test_generator[0]

tp = 0
wrong = [0, 0, 0]
total_iters = total // batch_size
for i in range(total_iters):
    print('正在评估第 {}/{} 个循环'.format(i, total_iters))
    data = next(test_generator)
    test_imgs = data[0]
    true_labels = data[1]
    filepaths = data[2]

    result = model.predict_on_batch(test_imgs)

    # 输出
    pred_labels = np.argmax(result, axis=-1)

    # 224,224,3
    single_img = test_imgs[0]
    single_path = filepaths[0]

    # 评估
    for k,v in enumerate(true_labels):

        print('true:{}, pred:{}, prob:{}, filepath:{}'.format(true_labels[k], pred_labels[k], result[k], filepaths[k]))

        # 输出cam
        draw(model, model_name, last_layer_name, len(classes), (img_height, img_width, 3), filepaths[k], filepaths[k].replace('mytest', 'mytest_show'))

        if true_labels[k] == pred_labels[k]:
            tp += 1
        else:
            wrong[true_labels[k]] += 1

print('total: {}, acc: {:.3f}'.format(total, tp*1.0/total))
print('wrong: {}'.format(wrong))

