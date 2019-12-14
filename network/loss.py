#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import tensorflow as tf
import keras.backend as K
from tensorflow.python.ops import array_ops
# import numpy as np

# 2.0 0.25
def focal_loss(gamma=2., alpha=0.25, num_classes=3, smoothing=None):

    def focal_loss_fixed(y_true, y_pred):
        """
        focal loss for multi-class classification
        fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
        :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
        :param y_pred: prediction after softmax shape of [batch_size, nb_class]
        :param alpha:
        :param gamma:
        :return:
        """
        # # parameters
        # alpha = 0.25
        # gamma = 2

        # To avoid divided by zero
        y_pred += K.epsilon()

        # smooth
        if smoothing:
            y_true -= smoothing * (y_true - 1. / tf.cast(num_classes, y_true.dtype))

        # Cross entropy
        ce = -y_true * tf.log(y_pred)

        # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
        # but refer to the definition of p_t, we do it
        weight = (1 - y_pred) ** gamma * y_true

        # Now fl has a shape of [batch_size, nb_class]
        # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
        # (CE has set unconcerned index to zero)
        #
        # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
        fl = ce * weight * alpha

        # Both reduce_sum and reduce_max are ok
        reduce_fl = K.max(fl, axis=-1)
        return reduce_fl

    return focal_loss_fixed

def focal_loss2(classes_num, gamma=5., alpha=.5, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed