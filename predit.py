# _*_ encoding: utf-8 _*_

import numpy as np
from data_gen import *
from model import Deeplabv3
import matplotlib.pyplot as plt
from loss import *

test_path = '/home/zhounan/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/img/'

print('load model...')
model = Deeplabv3(input_shape=(512, 512, 3), classes=2, backbone='mobilenetv2')
model.load_weights('checkpoints/DeepLabV3+-Weights-120.hdf5', by_name=True)
print('load model done.')

testGene = testGenerator(test_path)
results = model.predict_generator(testGene, 335, verbose=1)
masks = readmask()
# print(results.shape)
# print(masks.shape)

score = dice_coef(masks, results)
print(score)








