#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras.models import Model
from keras.layers import Input, GlobalAvgPool2D, Dense
from keras.applications import ResNet50
from data_loader import build_generator

def inference():

    # 载入模型
    img_width, img_height = 448,448
    input_tensor = Input(shape=(img_width, img_height, 3)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
    model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(200, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)
    weights_path = './model/bcnn_resnet50.h5'
    model2.load_weights(weights_path, by_name=True)

    # 读取数据
    root_path = '/home/speciallan/Documents/python/data/CUB_200_2011'
    train_dir = root_path + '/splitted/train'
    valid_dir = root_path + '/splitted/valid'
    batch_size = 16
    train_generator, valid_generator = build_generator(
        train_dir=train_dir,
        valid_dir=valid_dir,
        batch_size=batch_size)

    # 预测


if __name__ == '__main__':
    inference()
