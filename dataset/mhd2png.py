# -*- coding: utf-8 -*- 
import numpy as np
np.set_printoptions(threshold=np.inf)
import SimpleITK as itk
import os
from skimage import io
import tqdm
import cv2


def read_mhd_image(file_path):
    header = itk.ReadImage(file_path)
    image = np.array(itk.GetArrayFromImage(header))
    return image

def normalize_hu(image):
    '''
    将输入图像的像素值(-4000 ~ 4000)归一化到0~255之间
    '''
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image * 255
    return image

def mhd2png(img_mhd_dir, mask_mhd_dir, image_png_dir, mask_png_dir):
    for fileName in tqdm.tqdm(os.listdir(img_mhd_dir)):
        if os.path.splitext(fileName)[1] == '.mhd':
            case_id = os.path.splitext(fileName)[0]
            print(case_id)
            mask_dir = os.path.join(mask_mhd_dir, case_id + '.mhd')
            # print(mask_dir)
            if not os.path.exists(mask_dir):
                continue
            img_dir = os.path.join(img_mhd_dir, case_id + '.mhd')
            img = read_mhd_image(img_dir)
            img = normalize_hu(img)
            shape = list(np.shape(img))
            print(np.shape(img))
            mask = read_mhd_image(mask_dir)
            for index in range(len(img)):
                image_save_dir = os.path.join(image_png_dir, case_id + '-' + str(index) + '.png')
                mask_save_dir = os.path.join(mask_png_dir, case_id + '-' + str(index) + '.png')
                rgb = np.zeros((shape[1], shape[2], 3))
                mask_png = np.zeros((shape[1], shape[2], 3))
                # sum = np.sum(mask[index, :, :])
                if np.sum(mask[index, :, :]) != 0:
                    if index == 0 or index == len(img) - 1:
                        rgb[:, :, 0] = img[index, :, :]
                        rgb[:, :, 1] = img[index, :, :]
                        rgb[:, :, 2] = img[index, :, :]
                        mask_png[:, :, 0] = mask[index, :, :]
                        mask_png[:, :, 1] = mask[index, :, :]
                        mask_png[:, :, 2] = mask[index, :, :]
                    else:
                        rgb[:, :, 0] = img[index-1, :, :]
                        rgb[:, :, 1] = img[index, :, :]
                        rgb[:, :, 2] = img[index+1, :, :]
                        mask_png[:, :, 0] = mask[index, :, :]
                        mask_png[:, :, 1] = mask[index, :, :]
                        mask_png[:, :, 2] = mask[index, :, :]
                    png = rgb.astype(np.uint8)
                    mask_png = mask_png.astype(np.uint8)
                    png = cv2.resize(png, (512, 512), interpolation=cv2.INTER_AREA)
                    mask_png = cv2.resize(mask_png, (512, 512), interpolation=cv2.INTER_AREA)
                    io.imsave(image_save_dir, png)
                    io.imsave(mask_save_dir, mask_png)




if __name__ == '__main__':
    img_path = '/home/zhounan/Documents/dataset/tianchi/train_mhd/'
    mask_path = '/home/zhounan/Documents/dataset/tianchi/annotation'
    image_png_dir = '/home/zhounan/Documents/dataset/tianchi/img'
    mask_png_dir = '/home/zhounan/Documents/dataset/tianchi/mask'
    path1 = '/home/zhounan/Documents/dataset/tianchi/train_mhd/513670.mhd'
    path2 = '/home/zhounan/Documents/dataset/tianchi/annotation/513670.mhd'
    img = read_mhd_image(path1)
    print(np.shape(img))
    # mask = read_mhd_image(path2)
    # # print(np.shape(mask))
    # # print(len(img))
    # img = normalize_hu(img)
    # mhd2png(img_path, mask_path, image_png_dir, mask_png_dir)
    # png = np.zeros((768, 768, 3))
    # index = 4
    # png[:, :, 0] = img[index - 1, :, :]
    # png[:, :, 1] = img[index, :, :]
    # png[:, :, 2] = img[index + 1, :, :]
    # image = png.astype(np.uint8)
    # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    # save_dir = os.path.join(png_dir, '513670' + '-' + str(index) + '.png')
    # print(save_dir)
    # io.imsave(save_dir, image)