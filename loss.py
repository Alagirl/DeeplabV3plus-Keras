# -*- coding: utf-8 -*- 
import numpy as np
import keras.backend as K
np.set_printoptions(threshold=np.inf)

smooth=1.
def dice_coef(y_true, y_pred):
    # print(np.shape(y_true))
    # print(np.shape(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def weight_by_class_balance(truth, classes=[0, 1], class_weights=[0.75, 0.25]):
    weight_map = K.zeros_like(truth)
    for c, w in zip(classes, class_weights):
        class_mask = K.cast(K.equal(truth, c), 'float32')
        class_weight = w / (K.sum(class_mask) + K.epsilon())
        weight_map += (class_mask * class_weight)

    return weight_map


def weighted_log_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    weight_map = weight_by_class_balance(y_true_f)

    return -K.sum((y_true_f * K.log(y_pred_f + K.epsilon()) +
                   (1 - y_true_f) * K.log(1 - y_pred_f + K.epsilon())) * weight_map) / K.cast(K.shape(y_true)[0], 'float32')

def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    return -K.mean(y_true_f * ((1 - y_pred_f) ** 2) * K.log(y_pred_f + K.epsilon()) +
                  (1 - y_true_f) * (y_pred_f ** 2) * K.log(1 - y_pred_f + K.epsilon()))

def IoU_fun(eps=1e-6):
    def IoU(y_true, y_pred):
        # if np.max(y_true) == 0.0:
        #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        #
        ious = K.mean((intersection + eps) / (union + eps), axis=0)
        return K.mean(ious)

    return IoU

def IoU_loss_fun(eps=1e-6):
    def IoU_loss(y_true, y_pred):
        return 1 - IoU_fun(eps=eps)(y_true=y_true, y_pred=y_pred)


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
