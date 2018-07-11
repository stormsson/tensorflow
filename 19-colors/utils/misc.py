#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import keras.backend as K

import os
import yaml
from PIL import Image

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

def weighted_loss(y_true, y_pred):
    sq_loss = K.square(y_pred - y_true)
    return K.mean(sq_loss + (sq_loss * K.abs(y_true)), axis=-1)

def delete_not_256_images(folder):
    cnt = 0
    for filename in os.listdir(folder):
        with Image.open(folder+"/"+filename) as im:
            w,h = im.size

        if w !=256 or h != 256:
            os.remove(folder+"/"+filename)
            print ("deleted "+folder+"/"+filename+ "size: %dx%d" % (w,h))
            cnt+=1

def merged_discriminator_loss(generated_img, color_image_input):

    diff = l1_loss(color_image_input, generated_img)

    MULTIPLIER = 50

    def loss( y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + MULTIPLIER * diff


    return loss