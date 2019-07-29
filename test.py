# -*- coding: utf-8 -*- 
import tensorflow as tf
import keras.layers as KL
from keras.losses import mean_squared_error, binary_crossentropy
import keras.backend as K
import numpy as np
import cv2
import csv
import SimpleITK as sitk
from keras.utils import to_categorical
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from keras import *
from keras.layers import *
# path = '/home/zhounan/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/mask_gt/28-48.png'
# mask = cv2.imread(path)
# print(mask[mask>0])
# mask[mask > 0]=255
# plt.imshow(mask)
# plt.show()
# label = mask[:, :, 0]
# labels = to_categorical(label, num_classes=2)
# print(np.shape(np.where(labels[:,:,0]==0)))
#
# print(np.shape(np.where(labels[:,:,1]==0)))

# def SegNet():
#     model = Sequential()
#     #encoder
#     model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(3,256,256),padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
#     #(128,128)
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
#     #(64,64)
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
#     #(32,32)
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
#     #(16,16)
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
#     #(8,8)
#     #decoder
#     model.add(UpSampling2D(size=(2, 2)))
#     #(16,16)
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(UpSampling2D(size=(2, 2)))
#     #(32,32)
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(UpSampling2D(size=(2, 2)))
#     #(64,64)
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(UpSampling2D(size=(2, 2)))
#     #(128,128)
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(UpSampling2D(size=(2, 2)))
#     #(256,256)
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(4, (1, 1), strides=(1, 1), padding='same'))
#     model.add(Reshape((4, )))
#     #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)
#     model.add(Permute((2, 1)))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#     model.summary()
#     return model
#
# '''read csv as a list of lists'''
# def readCSV(filename):
#     lines = []
#     with open(filename, "rb") as f:
#         csvreader = csv.reader(f)
#         for line in csvreader:
#             lines.append(line)
#     return lines
# '''convert world coordinate to real coordinate'''
# def worldToVoxelCoord(worldCoord, origin, spacing):
#     stretchedVoxelCoord = np.absolute(worldCoord - origin)
#     voxelCoord = stretchedVoxelCoord / spacing
#     return voxelCoord
#
# # 加载结节标注
# anno_path = '/home/dataset/medical/luna16/CSVFILES/annotations.csv'
# annos = readCSV(anno_path)  # 共1186个结节标注
# print(len(annos))
# print(annos[0:3])
# '''
# 1187
# [['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm'], ['1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860', '-128.6994211', '-175.3192718', '-298.3875064', '5.651470635'], ['1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860', '103.7836509', '-211.9251487', '-227.12125', '4.224708481']]
# '''
#
# # 获取一个结节标注
# cand = annos[24]
# print(cand)
# '''
# ['1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492',
#  '-100.5679445',
#  '67.26051683',
#  '-231.816619',
#  '6.440878725']
# '''
# def load_itk_image(filename):
#     itkimage = sitk.ReadImage(filename)
#     numpyImage = sitk.GetArrayFromImage(itkimage)
#     numpyOrigin = np.array(list(itkimage.GetOrigin()))  # CT原点坐标
#     numpySpacing = np.array(list(itkimage.GetSpacing()))  # CT像素间隔
#     return numpyImage, numpyOrigin, numpySpacing
#
# numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
# # 将世界坐标下肺结节标注转换为真实坐标系下的坐标标注
# worldCoord = np.asarray([float(cand[1]),float(cand[2]),float(cand[3])])
# voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
# print(voxelCoord)

import cv2
import matplotlib.pyplot as pyplot


path = '/home/zhounan/Documents/dataset/tianchi/mask/318818-5.png'
img = cv2.imread(path)
img = img*64
pyplot.imshow(img)
pyplot.show()