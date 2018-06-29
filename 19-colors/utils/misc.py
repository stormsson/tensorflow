#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import keras.backend as K

"""
generate a fake image
"""
def generate_noise(amount=1, size=256, channels=3 ):
    noise = np.random.uniform(0, 1, (amount, size, size, channels))
    return noise

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def l2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def set_trainable(model, status):
    for layer in model.layers:
        layer.trainable = status
