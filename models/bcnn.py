#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Lambda
from keras.layers import Dropout, Flatten, Dense, GlobalAvgPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras import backend as K
from BCNN.models.xception import Xception


def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])


def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)


def l2_norm(x):
    return K.l2_normalize(x, axis=-1)


def bilinear_cnn(size, class_num, attention_module=None):
    input_tensor = Input(shape=(size[0], size[1], 3))
    model = Xception(include_top=False, weights=None,
                        input_tensor=input_tensor, attention_module=attention_module)
    cnn_out_a = model.layers[-2].output
    cnn_out_shape = model.layers[-2].output_shape
    cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2],
                         cnn_out_shape[-1]])(cnn_out_a)
    cnn_out_b = cnn_out_a
    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    cnn_out_dot = Reshape([cnn_out_shape[-1] * cnn_out_shape[-1]])(cnn_out_dot)

    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    output = Dense(class_num, activation='softmax')(l2_norm_out)
    model2 = Model(input_tensor, output)

    return model2