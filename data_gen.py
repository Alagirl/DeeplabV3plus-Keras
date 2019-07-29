# _*_ encoding: utf-8 _*_

from __future__ import print_function
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import skimage.io as io
import skimage.transform as trans
import itertools
import math
import keras
from keras.utils import to_categorical
import tensorflow as tf

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

filelist = []

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)

def trainGenerator(batch_size, data_dir, tf_session, tf_graph):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    # image_datagen = ImageDataGenerator(**aug_dict)
    # mask_datagen = ImageDataGenerator(**aug_dict)
    # image_generator = image_datagen.flow_from_directory(
    #     image_path,
    #     # classes = [image_folder],
    #     class_mode = None,
    #     color_mode = image_color_mode,
    #     target_size = target_size,
    #     batch_size = batch_size,
    #     save_to_dir = save_to_dir,
    #     save_prefix  = image_save_prefix,
    #     seed = seed)
    # mask_generator = mask_datagen.flow_from_directory(
    #     mask_path,
    #     # classes=[mask_folder],
    #     class_mode=None,
    #     color_mode=mask_color_mode,
    #     target_size=target_size,
    #     batch_size=batch_size,
    #     save_to_dir=save_to_dir,
    #     save_prefix=mask_save_prefix,
    #     seed=seed)
    # # train_generator = itertools.izip(image_generator, mask_generator)
    # train_generator = zip(image_generator, mask_generator)
    # for (img, mask) in train_generator:
    #     # print(np.shape(img))
    #     img, mask = adjustData(img, mask, flag_multi_class, num_class)

    image_path = os.path.join(data_dir, 'img')  # 获取path里面所有图片的路径
    img_list = os.listdir(image_path)
    steps = math.ceil(len(img_list) / batch_size)  # 确定每轮有多少个batch
    print("Found %s images." % len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: i * batch_size + batch_size]
            img = np.zeros(shape=(batch_size, 512, 512, 3), dtype=np.float)
            mask = np.zeros(shape=(batch_size, 512, 512, 5), dtype=np.float)
            j = -1
            for file in batch_list:
                j = j+1
                x = cv2.imread(file)
                img[j, :, :, :] = x
                # img[j,:,:,:] = cv2.resize(x, (512, 512), interpolation=cv2.INTER_CUBIC)
                # file.replace("jpg", "png")
                # file.replace("img", "mask")
                mask_path = os.path.join(data_dir, 'mask', file)
                y = cv2.imread(mask_path)
                y = y[:, :, 0]
                # y[np.where(y != 0)] = 1
                labels = to_categorical(y, num_classes=5)
                mask[j, :, :, :] = labels
                # mask[j,:,:,0] = cv2.resize(y, (512, 512), interpolation=cv2.INTER_CUBIC)
            # tf.summary.image('input/images', tf.cast(img, tf.uint8), max_outputs=3)
            # tf.summary.image('input/masks', tf.cast(mask*64, tf.uint8), max_outputs=3)
            # img = tf.cast(img, tf.float32)
            # img = img.eval()
            yield img, mask  # 把制作好的x, y生成出

def testGenerator(test_path, target_size = (512,512), flag_multi_class = False, as_gray = False):
    for filename in os.listdir(test_path):
        # print(filename)
        filelist.append(filename)
        img = io.imread(os.path.join(test_path, filename), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

def readmask():
    mask_path = '/home/zhounan/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/mask_gt/'
    masks = []
    for filename in filelist:
        mask = io.imread(os.path.join(mask_path, filename), as_gray=True)
        masks.append(to_categorical(mask, num_classes=2))
    masks = np.array(masks)
    return masks

def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2,
                 image_prefix="image", mask_prefix="mask", image_as_gray=True, mask_as_gray=True):
    '''
    该函数主要是分别在训练集文件夹下和标签文件夹下搜索图片，然后扩展一个维度后以array的形式返回，
    是为了在没用数据增强时的读取文件夹内自带的数据
    '''
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                         as_gray=mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr

def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    # 为不同类别涂上不同的颜色，color_dict[i]是与类别数有关的颜色，img_out[img == i,:]是img_out在img中等于i类的位置上的点
    return img_out / 255

def saveResult(save_path, npyfile, flag_multi_class = False, num_class = 2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img[img < 0.5] = 0
        img[img >= 0.5] = 255
        io.imsave(os.path.join(save_path, "%s" % filelist[i]), img)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    gen = data_generator(x_train, y_train, batch_size=4)
    x, y, l = next(gen)
    
    for i in range(len(x)):
        img = x[i]
        img.shape=(28,28,3)
        mask = y[i]
        mask.shape=(28,28)
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(mask,cmap='gray')
        plt.show()
    
    
    
    

