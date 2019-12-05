#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAvgPool2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.models import Model
from keras.applications import VGG16, ResNet50, InceptionV3
from BCNN.models.resnet_v1 import resnet_v1
from taurus_cv.models.resnet.snet import snettz

img_width = 224
img_height = 224
img_channel = 3

num_classes = 3

def snet(shape=(img_height, img_width, img_channel)):

    input_tensor = Input(shape=shape)
    model = snettz(input_tensor, classes_num=num_classes, is_extractor=True)
    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)
    return Model(inputs=input_tensor, outputs=x)


def resnet50(shape=(img_height, img_width, img_channel)):

    input_tensor = Input(shape=shape)
    model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def resnet20_se(shape=(img_height, img_width, img_channel)):
    depth = 20
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet50_se(shape=(img_height, img_width, img_channel)):
    depth = 50
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet101_se(shape=(img_height, img_width, img_channel)):
    depth = 104
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def cbam(shape=(img_height, img_width, img_channel)):
    depth = 50
    attention_module = 'cbam_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model
