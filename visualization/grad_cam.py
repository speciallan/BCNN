#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import keras
import cv2
import numpy as np
import keras.backend as K

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

K.set_learning_phase(1)  # set learning phase

def draw(model, model_name, last_layer_name, num_classes, shape, img_path, save_path):

    img_height, img_width, _ = shape

    # img=cv2.imread(img_path)

    # 这里居然是宽高
    image = load_img(img_path, target_size=(img_height, img_width))
    x = img_to_array(image)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)

    # 置信度最大类别
    class_idx = np.argmax(pred[0])

    # for i in range(num_classes):
    # class_idx = i
    class_output = model.output[:, class_idx]

    # block5
    last_conv_layer = model.get_layer(last_layer_name)
    total_channels = last_conv_layer.output.shape[-1]

    # print(len(K.gradients(class_output, last_conv_layer.output)))
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])
    # 512
    for i in range(total_channels):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(img_width, img_height), interpolation=cv2.INTER_NEAREST)

    # img = img_to_array(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # print(img.shape, heatmap.shape)
    # print(type(img), type(heatmap))
    # exit()
    superimposed_img = img.copy()
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0, superimposed_img, img.shape[2])

    cv2.imwrite(save_path.replace('.jpg', '_{}.jpg'.format(model_name)), superimposed_img)
    # cv2.imwrite(save_path.replace('.jpg', '_{}.jpg'.format(class_idx)), superimposed_img)
    # cv2.imshow('Grad-cam', superimposed_img)
    # cv2.waitKey(0)
