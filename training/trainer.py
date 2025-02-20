#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

"""
核心训练器
"""

import os
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def set_runtime_environment():
    """
    GPU设置，设置后端，包括字符精度
    :return:
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True  # 不要启动的时候占满gpu显存，按需申请空间
    session = tf.Session(config=cfg)     # 生成tf.session
    keras.backend.set_session(session)   # 设置后端为tensorflow
    # keras.backend.set_floatx('float16')  # 设置字符精度，默认float32，使用float16会提高训练效率，但是可能导致精度不够，梯度出现问题。


def get_callback(model_path, model_name, log_path='./logs', period=3):
    """
    定义callback，用于每个epoch回调
    包括模型检查点ModelCheckpoint,学习率自减少ReduceLROnPlateau,训练过程可视化TensorBoard，他们都继承自Callback
    :return:
    """
    checkpoint = ModelCheckpoint(filepath=os.path.join(model_path, model_name
                                          + "_e_{epoch:02d}"
                                          + "_loss_{val_loss:.3f}"
                                          + "_acc_{val_acc:.3f}"
                                          + ".h5"),                                  # 模型路径，默认保存在/tmp下
                                 monitor='val_acc',                                 # 监视值，包括精度acc、损失loss
                                 mode='max',
                                 verbose=1,                                          # 是否显示进度条
                                 save_best_only=True,                               # 知否只保存最好模型
                                 period=period)                                           # checkpoint间隔的epoch数量

    # 验证误差没有提升
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', # 监视值
                                   factor=0.5,     # 减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                   cooldown=2,     # 学习率减少后，会经过cooldown个epoch才重新进行检查
                                   patience=5,     # 经过patience个epoch后，如果检测值没变化，则出发学习率减少
                                   min_lr=1e-7,
                                   verbose=1,
                                   mode='auto')    # 最小学习率

    # 能够收敛，这里没使用早停
    early_stopping = EarlyStopping(monitor='val_loss', # 监视值
                                   patience=0,     # 早停出发后，经过patience个epoch停止训练
                                   verbose=1,      # 展示信息
                                   mode='auto')    # auto,min,max 当监测值不再减小/增加后触发早停

    # 保存训练日志
    logs = TensorBoard(log_dir=log_path)    # 日志保存路径，这里的值来自experiments里面的config.ini

    return [checkpoint, lr_reducer, logs]   # 继承自Callback的类都能保存成list返回


def train_retinanet(model, generator, epochs, iterations, init_epochs, init_weight_path, config):
    """
    训练rcnn(rpn+roiHead)
    :param model:
    :param generator:
    :param epochs:
    :param iterations:
    :param init_epochs:
    :param init_weight_path
    :return:
    """

    # 加载frcnn预训练模型
    if init_epochs > 0:
        model.load_weights(init_weight_path, by_name=True)

    # 有rpn预训练模型就加载，没有直接加载resnet预训练模型
    elif os.path.exists(config.rpn_weights):
        model.load_weights(config.rpn_weights, by_name=True)

    # 加载resnet预训练模型 或者自己的分类器权重
    else:
        model.load_weights(config.pretrained_weights, by_name=True)

    # 通过keras训练模型
    model.fit_generator(generator,
                        epochs=epochs,
                        steps_per_epoch=iterations,
                        verbose=1,
                        initial_epoch=init_epochs,
                        callbacks=get_callback('rcnn', config=config))

    return model
