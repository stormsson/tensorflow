#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, BatchNormalization
from keras.layers import LeakyReLU

import keras

import tensorflow as tf

from bcolors import bcolors
from os.path import isfile

from utils.misc import l1_loss

INPUT_SHAPE = (256, 256, 1,)

def build_GAN_generator(depth=10):
    model = Sequential()

    # input = 256x256 images, 3 channels (LAB, not RGB)

    model.add(Conv2D(64, (3, 3),
        activation='relu',
        padding='same',
        input_shape=INPUT_SHAPE ))

    # [depth] Conv2D Layers
    for x in xrange(1, depth-1):
        model.add(Conv2D(64, (3, 3),
            activation='relu',
            padding='same'))

        #add batchnorm for each layer except first and last
        model.add(BatchNormalization(momentum=0.9))
    model.add(Conv2D(3, (3, 3),
        activation='relu',
        padding='same'))

    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    model.compile(optimizer=optim, loss=l1_loss)

    return model

def build_GAN_discriminator():
    model = Sequential()

    D_INPUT_SHAPE = (256, 256, 3,)

    model.add(Conv2D(64, (4, 4),
        strides=(2, 2),
        padding='same',
        input_shape=D_INPUT_SHAPE))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(256, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(1, (4, 4),
        strides=(2, 2),
        padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))


    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=["accuracy"])

    return model


def get_gan_optimizer(LR=0.0002, B1=0.5, DECAY=0.0):

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    return optim