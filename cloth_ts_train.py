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
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical
from data_loader import build_generator, generator
from BCNN.training.trainer import set_runtime_environment, get_callback
from BCNN.models import model_zoo
from BCNN.network.loss import focal_loss

set_runtime_environment()

# 要修改
img_width, img_height = 224,224
img_width, img_height = 224,40
num_classes = 2

input_tensor = Input(shape=(img_height, img_width, 3))

# resnet50
# model = model_zoo.snet(shape=(img_height, img_width, 3))
# model = model_zoo.inception_resnet_v2(shape=(img_height, img_width, 3))
# model = model_zoo.resnet50(shape=(img_height, img_width, 3), num_classes=num_classes)
# model = model_zoo.resnet50_se(shape=(img_height, img_width, 3), num_classes=num_classes)
# model = model_zoo.resnet101(shape=(img_height, img_width, 3))
# model = model_zoo.cbam(shape=(img_height, img_width, 3))
# model = model_zoo.resnet20_se(shape=(img_height, img_width, 3))
# model = model_zoo.resnet32_se(shape=(img_height, img_width, 3))
# model = model_zoo.resnet32_se(shape=(img_height, img_width, 3))
# model = model_zoo.resnet38_se(shape=(img_height, img_width, 3))
model = model_zoo.resnet101_se(shape=(img_height, img_width, 3), num_classes=num_classes)
# model = model_zoo.resnet152_se(shape=(img_height, img_width, 3))

# for layer in model.layers[:]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
#     layer.trainable = True

# weights_path = '../taurus_cv/pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# weights_path = './model/cloth_resnet101_se.h5'
# weights_path = './model/cloth_resnet50_se.h5'
# weights_path = './model/cloth_ts_resnet50_2.h5'
# model.load_weights(weights_path, by_name=True, skip_mismatch=True)

model.summary()
# exit()

model.compile(loss=focal_loss(),
              optimizer=SGD(lr=1e-3,momentum=0.9),
              metrics=['accuracy']
              )

train_data_dir = '../../data/cloth/splitted/train_1'
train_data_dir = '../../data/cloth/train_2'
validation_data_dir = '../../data/cloth/test/test_2'
nb_train_samples = 65208
# nb_train_samples = 25712
# nb_train_samples = 5712
# nb_validation_samples = 6428
# nb_validation_samples = 3257
nb_validation_samples = 2527 #1
nb_validation_samples = 1035 #2
epochs = 200
batch_size = 128 # resnet50#64 101#128 152#48
classes = ['02', '99']
model_path = './model'
model_name = 'cloth_resnet50_se'

train_generator, valid_generator = build_generator(
    train_dir=train_data_dir,
    valid_dir=validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

# (16, 224, 224, 3) , (16, 3)
# t = next(train_generator)
# print(t[0].shape, t[0])
# exit()

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=get_callback(model_path, model_name, period=1))

# model2.save_weights(model_path + '/' + model_name + '.h5')

# -------------------------------------------------------------------
exit()







train_generator = generator(train_data_dir, classes=classes, batch_size=batch_size, target_size=(img_height, img_width))

path_model = './model/resnet50'
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
                            , patience=5
                            , min_delta=1e-3
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
    ,callback_stopper
]

for e in range(epochs):
    next_flag = True
    i = 0
    tp = 0

    while next_flag:

        imgs, labels, filepaths, next_flag = next(train_generator)
        labels_one_hot = to_categorical(labels, num_classes)
        hist = model2.train_on_batch(imgs, labels_one_hot)
        i += len(imgs)

        result = model2.predict_on_batch(imgs)
        pred_labels = np.argmax(result, axis=-1)
        for k,v in enumerate(labels):
            if labels[k] == pred_labels[k]:
                tp += 1

        print('正在训练第 Epoch {}/{}, {} 条数据, acc:{}'.format(e+1, epochs, i, tp*1.0/i))

        if i > total:
            next_flag = False

    # 跑验证集
    print('正在训练第 Epoch {}/{}, {} 条数据, acc:{}'.format(e+1, epochs, i, tp*1.0/i))

model2.save_weights('model/cloth_resnet50.h5')



