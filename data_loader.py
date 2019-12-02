'''Load data and build generators.'''

from data_preprocesser import normalize_image, random_crop_image, center_crop_image
from data_preprocesser import resize_image, horizontal_flip_image
from data_preprocesser import ImageDataGenerator
import numpy as np
import cv2
import random
import os

def train_preprocessing(x, size_target=(224, 224)):
    '''Preprocessing for train dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return horizontal_flip_image(
            resize_image(
                x,
                size_target=size_target,
                flg_keep_aspect=True
            )
    )

    return normalize_image(
        random_crop_image(
            horizontal_flip_image(
                resize_image(
                    x,
                    size_target=size_target,
                    flg_keep_aspect=True
                )
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def valid_preprocessing(x, size_target=(224, 224)):
    '''Preprocessing for validation dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return resize_image(
            x,
            size_target=size_target,
            flg_keep_aspect=True
    )

    return normalize_image(
        center_crop_image(
            resize_image(
                x,
                size_target=size_target,
                flg_keep_aspect=True
            )
        ),
        # mean=[123.82988033, 127.3509729, 110.25606303]
    )

def build_generator(
        train_dir=None,
        valid_dir=None,
        target_size=(224,224),
        batch_size=128
    ):
    '''Build train and validation dataset generators.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        batch_size: batch size.

    Returns:
        Train generator and validation generator.
    '''
    results = []
    if train_dir:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=train_preprocessing
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [train_generator]

    if valid_dir:
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=valid_preprocessing
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [valid_generator]

    return results

def generator(dir, classes, batch_size=16, target_size=(256,256)):

    class_list = os.listdir(dir)
    label_list, filepath_list = [], []

    # 反转
    classes_dict = {v:k for k,v in enumerate(classes)}

    for class_name in class_list:

        file_list = os.listdir(os.path.join(dir, class_name))

        for filename in file_list:

            filepath = os.path.join(dir, class_name, filename)
            label_list.append(classes_dict[class_name])
            filepath_list.append(filepath)

    length = len(filepath_list)
    idx_list = [i for i in range(length)]
    random.shuffle(idx_list)

    label_list = np.array(label_list)
    filepath_list = np.array(filepath_list)

    idx_batches = [idx_list[k:k + batch_size] for k in range(0, len(idx_list), batch_size)]
    xb_length = len(idx_batches)
    i = 0
    next_flag = True

    while True:

        if i == xb_length - 1:
            next_flag = False

        if i == xb_length:
            i = 0
            next_flag = True

        idx_list = idx_batches[i]
        i += 1

        # 读取图片
        imgs, labels, filepaths = [], [], []

        for idx in idx_list:

            img = cv2.imread(filepath_list[idx])
            img_resized = cv2.resize(img, target_size)
            img_preprocessed = img_resized / 255.0

            imgs.append(img_preprocessed)
            labels.append(label_list[idx])
            filepaths.append(filepath_list[idx])

        imgs = np.array(imgs)
        labels = np.array(labels)
        filepaths = np.array(filepaths)

        yield imgs, labels, filepaths, next_flag


def gen1():
    x =np.array(['a','b','c','d','e','f','g'])
    random.shuffle(x)
    batch_size = 2

    length = len(x)
    idx_list = [i for i in range(length)]
    random.shuffle(idx_list)

    idx_batches = [idx_list[k:k + batch_size] for k in range(0, len(idx_list), batch_size)]
    xb_length = len(idx_batches)
    i = 0

    next_flag = True

    while True:
        if i == xb_length-1:
            next_flag = False
        if i == xb_length:
            i = 0
            next_flag = True

        idx_list = idx_batches[i]
        i += 1
        yield x[idx_list], next_flag


if __name__ == "__main__":
    gen = gen1()
    print(next(gen))
    print(next(gen))
    print(next(gen))
    print(next(gen))
    print(next(gen))

