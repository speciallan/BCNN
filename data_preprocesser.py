'''Preprocessing data.'''

import os
import numpy as np
import cv2

from tensorflow.python.keras.preprocessing.image import DirectoryIterator as Keras_DirectoryIterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator as Keras_ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.python.keras.backend import floatx
import keras.backend as K
import random

np.random.seed(3)

def resize_image(
        x,
        size_target=(448, 448),
        rate_scale=1.0,
        flg_keep_aspect=False,
        flg_random_scale=False):
    '''Resizing image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.
        rate_scale: scale rate.
        flg_keep_aspect: a bool of keeping image aspect or not.
        flg_random_scale: a bool of scaling image randomly.

    Returns:
        Resized image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Calculate resize coefficients
    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c , = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c , = img.shape

    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]
    if size_target is None:
        size_heigth_target = size_height_img * rate_scale
        size_width_target = size_width_img * rate_scale

    coef_height, coef_width = 1, 1
    if size_height_img < size_heigth_target:
        coef_height = size_heigth_target / size_height_img
    if size_width_img < size_width_target:
        coef_width = size_width_target / size_width_img

    # Calculate coeffieient to match small size to target size
    ## scale coefficient if specified
    low_scale = rate_scale
    if flg_random_scale:
        low_scale = 1.0
    coef_max = max(coef_height, coef_width) * np.random.uniform(low=low_scale, high=rate_scale)

    # Resize image
    size_height_resize = np.ceil(size_height_img*coef_max)
    size_width_resize = np.ceil(size_width_img*coef_max)

    method_interpolation = cv2.INTER_CUBIC

    if flg_keep_aspect:
        img_resized = cv2.resize(
            img,
            dsize=(int(size_width_resize), int(size_height_resize)),
            interpolation=method_interpolation)
    else:
        img_resized = cv2.resize(
            img,
            dsize=(
                int(size_width_target*np.random.uniform(low=low_scale, high=rate_scale)),
                int(size_heigth_target*np.random.uniform(low=low_scale, high=rate_scale))),
            interpolation=method_interpolation)

    return img_resized

def center_crop_image(x, size_target=(448, 448)):
    '''Crop image from center point.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Center cropped image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c, = img.shape

    # Crop image
    h_start = int((size_height_img - size_heigth_target) / 2)
    w_start = int((size_width_img - size_width_target) / 2)
    img_cropped = img[h_start:h_start+size_heigth_target, w_start:w_start+size_width_target, :]

    return img_cropped

def random_crop_image(x, size_target=(448, 448)):
    '''Crop image from random point.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Random cropped image.
    '''

    # Convert to numpy array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_target) == 1:
        size_heigth_target = size_target
        size_width_target = size_target
    if len(size_target) == 2:
        size_heigth_target = size_target[0]
        size_width_target = size_target[1]

    if len(img.shape) == 4:
        _o, size_height_img, size_width_img, _c , = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        size_height_img, size_width_img, _c , = img.shape

    # Crop image
    margin_h = (size_height_img - size_heigth_target)
    margin_w = (size_width_img - size_width_target)
    h_start = 0
    w_start = 0
    if margin_h != 0:
        h_start = np.random.randint(low=0, high=margin_h)
    if margin_w != 0:
        w_start = np.random.randint(low=0, high=margin_w)
    img_cropped = img[h_start:h_start+size_heigth_target, w_start:w_start+size_width_target, :]

    return img_cropped

def horizontal_flip_image(x):
    '''Flip image horizontally.

    Args:
        x: input image.

    Returns:
        Horizontal flipped image.
    '''

    if np.random.random() >= 0.5:
        return x[:, ::-1, :]
    else:
        return x

def normalize_image(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):
    '''Normalization.

    Args:
        x: input image.
        mean: mean value of the input image.
        std: standard deviation value of the input image.

    Returns:
        Normalized image.
    '''

    x = np.asarray(x, dtype=np.float32)
    if len(x.shape) == 4:
        for dim in range(3):
            x[:, :, :, dim] = (x[:, :, :, dim] - mean[dim]) / std[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[:, :, dim] = (x[:, :, dim] - mean[dim]) / std[dim]

    return x

def preprocess_input(x):
    '''Preprocesses a tensor or Numpy array encoding a batch of images.'''

    return normalize_image(x, mean=[123.82988033, 127.3509729, 110.25606303])


class DirectoryIterator(Keras_DirectoryIterator):
    '''Inherit from keras' DirectoryIterator.'''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 follow_links=False, interpolation='nearest'):
        # 定义加载文件名list
        self.order_filenames = []
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))
        for res in results:
            classes, filenames = res.get()
            self.classes[i:i + len(classes)] = classes
            self.filenames += filenames
            i += len(classes)
        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            # 获取数据生成器加载的文件名
            self.order_filenames.append(fname)
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


    # def _get_batches_of_transformed_samples(self, index_array):
    #     batch_x = np.zeros(
    #         (len(index_array),) + self.image_shape,
    #         dtype=floatx())
    #     grayscale = self.color_mode == 'grayscale'
    #
    #     # Build batch of image data
    #     for i, j in enumerate(index_array):
    #         fname = self.filenames[j]
    #         img = load_img(
    #             os.path.join(self.directory, fname),
    #             grayscale=grayscale,
    #             target_size=None,
    #             interpolation=self.interpolation)
    #         x = img_to_array(img, data_format=self.data_format)
    #
    #         # Pillow images should be closed after `load_img`, but not PIL images.
    #         if hasattr(img, 'close'):
    #             img.close()
    #
    #         x = self.image_data_generator.standardize(x)
    #         batch_x[i] = x
    #
    #     # Optionally save augmented images to disk for debugging purposes
    #     if self.save_to_dir:
    #         for i, j in enumerate(index_array):
    #             img = array_to_img(batch_x[i], self.data_format, scale=True)
    #             fname = '{prefix}_{index}_{hash}.{format}'.format(
    #                 prefix=self.save_prefix,
    #                 index=j,
    #                 hash=np.random.randint(1e7),
    #                 format=self.save_format)
    #             img.save(os.path.join(self.save_to_dir, fname))
    #
    #     # Build batch of labels
    #     if self.class_mode == 'input':
    #         batch_y = batch_x.copy()
    #     elif self.class_mode == 'sparse':
    #         batch_y = self.classes[index_array]
    #     elif self.class_mode == 'binary':
    #         batch_y = self.classes[index_array].astype(floatx())
    #     elif self.class_mode == 'categorical':
    #         batch_y = np.zeros(
    #             (len(batch_x), self.num_classes),
    #             dtype=floatx())
    #         for i, label in enumerate(self.classes[index_array]):
    #             batch_y[i, label] = 1.
    #     else:
    #         return batch_x
    #
    #     return batch_x, batch_y


class ImageDataGenerator(Keras_ImageDataGenerator):
    '''Inherit from keras' ImageDataGenerator.'''
    def flow_from_directory(
            self, directory,
            target_size=(256, 256), color_mode='rgb',
            classes=None, class_mode='categorical',
            batch_size=16, shuffle=True, seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            follow_links=False,
            subset=None,
            interpolation='nearest'
        ):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)

