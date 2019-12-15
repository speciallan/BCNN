#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAvgPool2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input, Add
from keras.models import Model
from keras.applications import VGG16, ResNet50
import keras_resnet
from keras_resnet.models import ResNet101
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from BCNN.models.resnet_v1 import resnet_v1
from BCNN.models.xception import Xception
from BCNN.models.attention_module import attach_attention_module
from taurus_cv.models.resnet.snet import snettz

from keras.utils import get_file
from keras.layers import *
from keras import Model, layers, backend
# from keras.applications import imagenet_utils
# from keras.applications.imagenet_utils import imagenet_utils

from BCNN.models.efficientnet.efficientnet import EfficientNetB4, EfficientNetB5
from BCNN.models.efficientnet.custom_objects import EfficientNetDenseInitializer

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

def resnet50(shape=(img_height, img_width, img_channel), num_classes=num_classes):

    input_tensor = Input(shape=shape)
    model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def resnet101(shape=(img_height, img_width, img_channel)):

    backbone = 'resnet101'
    weights_path = download_imagenet(backbone)

    input_tensor = Input(shape=shape)
    model = keras_resnet.models.ResNet101(input_tensor, include_top=True, freeze_bn=True)
    model.load_weights(weights_path)

    x = model.get_layer('res5c_relu').output

    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def inception_resnet(shape=(img_height, img_width, img_channel)):

    input_tensor = Input(shape=shape)

    weights = 'imagenet' if shape[2] == 3 else None

    model = InceptionResNetV2(include_top=False, weights=weights, input_tensor=input_tensor)

    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def xception(shape=(img_height, img_width, img_channel), num_classes=num_classes):

    input_tensor = Input(shape=shape)

    weights = 'imagenet' if shape[2] == 3 else None

    model = Xception(include_top=False, weights=weights, input_tensor=input_tensor)

    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    # x = Flatten()(x)
    # x = Dense(100, activation='softmax', name='fc1')(x)
    # x = Dropout(rate=0.5)(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def efficientnet_b4(shape=(img_height, img_width, img_channel), num_classes=num_classes):

    input_tensor = Input(shape=shape)
    model = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=input_tensor)

    x = model.output
    x = GlobalAvgPool2D(name='avg_pool')(x)
    # x = layers.Dense(num_classes)(x)
    x = layers.Dense(num_classes, kernel_initializer=EfficientNetDenseInitializer())(x)

    model2 = Model(inputs=model.input, outputs=x)
    return model2

def resnet50_se(shape=(img_height, img_width, img_channel), num_classes=num_classes):

    input_tensor = Input(shape=shape)
    model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    x = model.output
    y = attach_attention_module(x, 'se_block')
    x = Add()([x, y])
    x = Activation('relu')(x)
    x = GlobalAvgPool2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='fc')(x)

    model2 = Model(inputs=model.input, outputs=x)

    return model2

def resnet50_se1(shape=(img_height, img_width, img_channel)):
    depth = 50
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet20_se(shape=(img_height, img_width, img_channel)):
    depth = 20
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet32_se(shape=(img_height, img_width, img_channel)):
    depth = 32
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet38_se(shape=(img_height, img_width, img_channel)):
    depth = 38
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet101_se(shape=(img_height, img_width, img_channel), num_classes=num_classes):
    depth = 104
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def resnet152_se(shape=(img_height, img_width, img_channel)):
    depth = 152
    attention_module = 'se_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

def cbam(shape=(img_height, img_width, img_channel)):
    depth = 50
    attention_module = 'cbam_block'
    model = resnet_v1(input_shape=shape, depth=depth, num_classes=num_classes, attention_module=attention_module)
    return model

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


def download_imagenet(backbone):
    filename = resnet_filename.format(backbone[6:])
    resource = resnet_resource.format(backbone[6:])
    if backbone == 'resnet50':
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 'resnet101':
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 'resnet152':
        checksum = '6ee11ef2b135592f8031058820bb9e71'
    else:
        raise ValueError("Il backbone '{}' non è riconosciuto.".format(backbone))

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


