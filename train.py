# _*_ encoding: utf-8 _*_

import os
import keras
from keras.callbacks import ModelCheckpoint, Callback
from callback import Tensorboard
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from model import Deeplabv3
from loss import dice_coef, dice_coef_loss
from data_gen import *
import keras.backend as K
from keras.callbacks import LearningRateScheduler

class ParallerModelCheckPoint(Callback):
    def __init__(self, single_model):
        self.mode_to_save = single_model
        
    def on_epoch_end(self, epoch, logs={}):
        print(r'save model: checkpoints/DeepLabV3+-Weights-%02d.hdf5'%(epoch+1))
        self.mode_to_save.save_weights(r'checkpoints/DeepLabV3+-Weights-%02d.hdf5'%(epoch+1))

# 设置使用的显存以及GPU
# 设置可用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# keras设置GPU参数
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.888
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# tf.keras.backend.set_session(session)
use_gpu=True
gpus=1
learning_rate=0.01
log_dir = '/home/zhounan/PycharmProjects/deeplab/DeeplabV3Plus-Keras-Retraining-master/tflog/'
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

K.clear_session()
# xception, mobilenetv2
basemodel = Deeplabv3(input_shape=(512, 512, 3), classes=5, backbone='mobilenetv2')
basemodel.summary()
K.tf.summary.image('output/pred', tf.cast(basemodel.output * 64., tf.uint8), max_outputs=3)

# model_file = 'checkpoints/DeepLabV3+-Weights-40.hdf5'
# if os.path.exists(model_file):
#     print('loading model:', model_file)
#     basemodel.load_weights(model_file, by_name=True)

if gpus > 1:
    parallermodel = multi_gpu_model(basemodel, gpus=gpus)
    checkpoint = ParallerModelCheckPoint(basemodel)
else:
    model = basemodel
    checkpoint = ModelCheckpoint(r'checkpoints/DeepLabV3+-Weights-{epoch:02d}.hdf5', save_weights_only=True,
                                 verbose=2, monitor='val_loss', period=1)
optimizer = SGD(lr=learning_rate, momentum=0.9, clipnorm=5.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.metrics_names.append('Dice')
# model.metrics_tensors.append(dice_coef)
data_dir = '/home/zhounan/Documents/dataset/tianchi/'
# mask_path = '/home/zhounan/Documents/dataset/tianchi/mask/'
tf_session = keras.backend.get_session()
tf_graph = tf.get_default_graph()
train_gen = trainGenerator(4, data_dir, tf_session, tf_graph)
image, mask = train_gen.__next__()
tf.summary.image('input/images', tf.cast(image, tf.uint8), max_outputs=3)
tf.summary.image('input/masks', tf.cast(mask*64, tf.uint8), max_outputs=3)
tf.summary.scalar('learning_rate', model.optimizer.lr)
summary_op = tf.summary.merge_all()
mytensorboard = Tensorboard(summary_op, batch_interval=10, log_dir=log_dir)
# earlystop = EarlyStopping(monitor='acc', patience=10, verbose=1)
print('begin training...')
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)
model.fit_generator(train_gen,
                        steps_per_epoch=3000,
                        epochs=120, 
                        verbose=1,
                        callbacks=[checkpoint, mytensorboard, reduce_lr],
                        initial_epoch=0)





