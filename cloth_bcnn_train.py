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
from keras.callbacks import EarlyStopping
from keras.models import load_model
from data_loader import build_generator, generator
from model_builder import build_bcnn


K.set_image_dim_ordering('tf')

WEIGHTS_PATH = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# img_width, img_height = 200,200
# img_width, img_height = 224,224
img_width, img_height = 224,224
num_classes = 3

model = build_bcnn(
    all_trainable=True,
    size_height=img_height,
    size_width=img_width,
    no_class=num_classes,
    # learning_rate=1e-3,
    learning_rate=1.0,
    decay_learning_rate=0.1,
    decay_weight_rate=1e-8
)

# model.load_weights('./model/cloth_bcnn.h5')

train_data_dir = '../../data/cloth/splitted/train'
valid_data_dir = '../../data/cloth/splitted/valid'
nb_train_samples = 25712
nb_validation_samples = 6428
epochs = 10
batch_size = 32
classes = ['01', '02', '99']

# train_generator, valid_generator = build_generator(
#     train_dir=train_data_dir,
#     valid_dir=validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size)

train_generator = generator(train_data_dir, classes=classes, batch_size=batch_size, target_size=(img_height, img_width))
valid_generator = generator(valid_data_dir, classes=classes, batch_size=batch_size, target_size=(img_height, img_width))

# (16, 224, 224, 3) , (16, 3)

# t = next(train_generator)
# print(t[0].shape, t[1].shape)
# exit()

path_model = './model/cloth_bcnn_'
callack_saver = keras.callbacks.ModelCheckpoint(
                        path_model
                            + "e_{epoch:02d}"
                            + "_loss_{val_loss:.3f}"
                            + "_acc_{val_acc:.3f}"
                            + ".h5"
                        , monitor='val_loss'
                        , verbose=0
                        , mode='auto'
                        , period=1
                        , save_best_only=True
                    )
callback_reducer = keras.callbacks.ReduceLROnPlateau(
                                monitor='val_loss'
                                , factor=0.1
                                , min_lr=1e-6
                                , min_delta=1e-3
                                , patience=5
                            )
callback_stopper = keras.callbacks.EarlyStopping(
                        monitor='val_loss'
                        , min_delta=1e-3
                        , patience=5
                        , verbose=0
                        , mode='auto'
                    )
list_callback = [
    callack_saver
    ,callback_reducer
    # ,callback_stopper
]



model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=list_callback)

model.save_weights('model/cloth_bcnn.h5')


