#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

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
from data_loader import build_generator


K.set_image_dim_ordering('tf')

WEIGHTS_PATH = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# img_width, img_height = 200,200
# img_width, img_height = 224,224
img_width, img_height = 448,448

#model = VGG16(include_top=False, weights='imagenet')

input_tensor = Input(shape=(img_width, img_height, 3)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
weights_path = '../taurus_cv/pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model.load_weights(weights_path, by_name=True)
print('Model loaded.')


x = model.output
x = GlobalAvgPool2D(name='avg_pool')(x)
x = Dense(200, activation='softmax', name='fc')(x)
# x = Dense(512,activation='relu')(x)
# x = Dense(512,activation='relu')(x)
# x = Dense(200, activation = 'softmax')(x)

model2 = Model(inputs=model.input, outputs=x)


#model2 = load_model('mstar.h5')


for layer in model2.layers[:]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
    layer.trainable = True

model2.compile(loss='categorical_crossentropy',
               optimizer = SGD(lr=1e-3,momentum=0.9),#SGD(lr=1e-3,momentum=0.9)
               metrics=['accuracy'])



train_data_dir = '../../data/CUB_200_2011/splitted/train'
validation_data_dir = '../../data/CUB_200_2011/splitted/valid'
#img_width, img_height = 128, 128
nb_train_samples = 5994
nb_validation_samples = 5794
epochs = 33
batch_size = 4

train_generator, valid_generator = build_generator(
    train_dir=train_data_dir,
    valid_dir=validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     # shear_range=0.2,
#     # rotation_range=10.,
#     # zoom_range=0.2,
#     # horizontal_flip=True
# )
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # 图片generator
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

#model2.load_weights('mstar.h5')

model2.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=nb_validation_samples // batch_size)

model2.save_weights('model/bcnn_resnet50.h5')


