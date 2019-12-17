############################
# 图片批量预处理工具
#1029编写，主要改进：针对过暗的图片进行亮度自适应调整
import cv2
import os
import numpy as np
from math import *
from numpy import *
from tqdm import tqdm

def extension(img, size):
    # 拓宽图片的边缘，
    # 输入：用于拓宽的图片（灰度图），以及需要达到的大小(高，宽)
    # 输出：拓宽之后的图片
    # COLOR = [int(np.mean(img[:, :, 0])), int(np.mean(img[:, :, 1])), int(np.mean(img[:, :, 2]))]
    if len(img.shape) == 3:
        width, height, channels = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # COLOR = [int(np.mean(img[:, :, 0])), int(np.mean(img[:, :, 1])), int(np.mean(img[:, :, 2]))]
    else:
        width, height = img.shape
        # COLOR = int(np.mean(img))

    # img = (0.5 * cos((mat(img) / 255.0 + 1) * pi) + 0.5) * 255

    detaX = size[0] - width
    detaY = size[1] - height
    if width > size[0]:
        width = size[0]
        detaX = 0
    if height > size[1]:
        height = size[1]
        detaY = 0

    res = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
    #若目标尺寸小于原始尺寸，将图片resize为目标尺寸
    top = int(detaX / 2)
    left = int(detaY / 2)
    #final = cv2.copyMakeBorder(res, top, detaX - top, left, detaY - left, cv2.BORDER_REPLICATE)
    B = max(np.mean(res[:11,:11,0]),np.mean(res[-11:,:11,0]),np.mean(res[:11,-11:,0]),np.mean(res[-11:,-11:,0]))
    G = max(np.mean(res[:11,:11,1]),np.mean(res[-11:,:11,1]),np.mean(res[:11,-11:,1]),np.mean(res[-11:,-11:,1]))
    R = max(np.mean(res[:11,:11,2]),np.mean(res[-11:,:11,2]),np.mean(res[:11,-11:,2]),np.mean(res[-11:,-11:,2]))
    final = cv2.copyMakeBorder(res, top, detaX - top, left, detaY - left,cv2.BORDER_CONSTANT,value=[B,G,R])
    #边界拓展，2-5个参数分别为上下左右需要拓展的尺寸
    #  cv2.BORDER_REPLICATE)  # cv2.BORDER_CONSTANT, value=COLOR)
    return final

def generatePic(img, length, height_target):
    if len(img.shape) == 3:
        height, width, channels = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width, = img.shape
    imgs = []
    for k in range(int(np.ceil(width / length))):
        from_idx = k * length
        to_idx = (k + 1) * length

        if width<length:
            imgs.append(extension(img[:, from_idx:to_idx,:], (height_target, length)))
        else:
            if to_idx<width: #原纤维较长，切割的整数部分
                imgs.append(extension(img[:, from_idx:to_idx,:], (height_target, length)))

            elif (width - from_idx) > 0.18 * length:  # 不够切割的部分
                imgs.append(extension(img[:, width-length:width,:], (height_target, length)))

    return imgs

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel

def aug(src):
    """图像亮度增强"""
    lightness, image_cut = get_lightness(src)
    # if lightness > 130:
    #     print("亮度足够，不做增强")
    # print("亮度不足，进行增强")
    #plt.hist(np.array(src).ravel(), 256, [0, 256])

    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(image_cut, 1, 99)
    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel
    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    return out,lightness


def get_lightness(src):
    # 计算亮度
    height, width, channel= src.shape
    image_cut = src[int(height*0.2):int(height*0.8),int(width*0.2):int(width*0.8),:]
    hsv_image = cv2.cvtColor(image_cut, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    return lightness, image_cut

if __name__ == '__main__':
    rootpath = '../../data/cloth/splitted/train/'     #原图大目录
    des = '../../data/cloth/splitted/train_320/'     #原图大目录

    rootpath = '../../data/cloth/test/test/'
    des = '../../data/cloth/test/test_320/'

    # rootpath = '../../data/cloth/test/test_c/'
    # des = '../../data/cloth/test/test_c_320/'

    for fibre in os.listdir(rootpath):
        # if fibre == '01':
        #     continue
        if os.path.exists(des + fibre):
            del_file(des + fibre)
            #print('不删除原文件夹')
        else:
            os.makedirs(des + fibre)
        for name in tqdm(os.listdir(rootpath+fibre)):
            #print(name)
            if name.endswith('.jpg'):
                try:
                    image = cv2.imread(rootpath+fibre+'/'+name) #读图

                    imgs = generatePic(image, 320, 80)
                    #第二个元素为切图slice宽度，第三个元素为图像扩展高度
                    for k in range(len(imgs)):
                        image = imgs[k]

                        # gray
                        # image, light = aug(image)
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # image = image[:,:,np.newaxis]

                        #存三通道or单通道
                        #image = np.concatenate((image, image, image), axis=2)
                        name1=name[:-4]
                        #print(name1)
                        cv2.imwrite(des +fibre+'/'+name1+'['+str(k)+']'+'.jpg', image)
                except:
                    continue