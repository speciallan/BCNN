#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings


from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend, layers, models
from keras import utils as keras_utils
from BCNN.models.attention_module import attach_attention_module


TF_WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/'
    'releases/download/v0.4/'
    'xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def Xception(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             attention_module=None,
             **kwargs):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet. This model can
    only be used with the data format `(width, height, channels)`.
    You should set `image_data_format='channels_last'` in your Keras config
    located at ~/.keras/keras.json.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if backend.image_data_format() != 'channels_last':
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" '
                      '(channels, width, height). '
                      'You should set `image_data_format="channels_last"` '
                      'in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=71,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        # attention_module
        if attention_module is not None:
            x = attach_attention_module(x, attention_module)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='xception')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = keras_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        backend.set_image_data_format(old_data_format)
    return model


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)
