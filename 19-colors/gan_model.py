#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, multiply, concatenate
from keras.layers import Conv2D, BatchNormalization
from keras.layers import LeakyReLU


import keras

import tensorflow as tf

from bcolors import bcolors
from os.path import isfile

from utils.misc import l1_loss, l2_loss, weighted_loss

INPUT_SHAPE = (256, 256, 3,)



def get_custom_objects_for_restoring():
    custom_objs = {
        "l1_loss": l1_loss,
        "l2_loss": l2_loss,
        "weighted_loss": weighted_loss
    }

    return custom_objs


def build_GAN_generator(depth=10, compiled=False):

    with tf.name_scope("generator_model"):
        generator_model = Sequential()

        # with tf.name_scope("GENERATOR_MODEL"):
        generator_model.add(Conv2D(64, kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            input_shape=INPUT_SHAPE,
            name="gen_conv2d_0" ))

        # [depth] Conv2D Layers
        for x in xrange(1, depth):
            generator_model.add(Conv2D(64, kernel_size=3,
                strides=1,
                activation='relu',
                padding='same',
                name="gen_conv2d_"+str(x)))

            #add batchnorm for each layer except first and last

            generator_model.add(BatchNormalization(momentum=0.9,
                name="gen_bn_"+str(x)))

        generator_model.add(Conv2D(3, kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            name="img_output"))


        noise_input = Input(shape=INPUT_SHAPE, name="gen_noise_input")
        bw_image_input = Input(shape=INPUT_SHAPE, name="gen_bw_input")

        with tf.name_scope("gen_input_mult"):
            model_input = multiply([noise_input, bw_image_input])

        img = generator_model(model_input)

        composed =  Model(inputs=[noise_input, bw_image_input], outputs=img)

    # LR = 0.0002
    # B1 = 0.5
    # DECAY = 0.0

    # optim = get_generator_optimizer()

    # losses = [ l2_loss ]
    # composed.compile(optimizer=optim, loss=losses, metrics=[ l1_loss ])

    return composed

def buid_LAB_generator(depth=10):

    LAB_INPUT_SHAPE = ( 256, 256, 1, )

    noise_a_input = Input(shape=LAB_INPUT_SHAPE, name="g_noise_for_a_chan_input")
    noise_b_input = Input(shape=LAB_INPUT_SHAPE, name="g_noise_for_b_chan_input")
    l_image_input = Input(shape=LAB_INPUT_SHAPE, name="g_l_chan_input")

    model_a_input = multiply([noise_a_input, l_image_input], name="g_a_chan_input_mult")
    model_b_input = multiply([noise_b_input, l_image_input], name="g_b_chan_input_mult")

    input_fusion = concatenate([model_a_input, model_b_input], axis=3, name="g_ab_chans_fusion")

    with tf.name_scope("generator_model"):
        generator_model = Sequential()

        # with tf.name_scope("GENERATOR_MODEL"):
        generator_model.add(Conv2D(64, kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            input_shape=(256, 256, 2, ),
            name="gen_conv2d_0" ))

        # [depth] Conv2D Layers
        for x in xrange(1, depth):
            generator_model.add(Conv2D(64, kernel_size=3,
                strides=1,
                activation='relu',
                padding='same',
                name="gen_conv2d_"+str(x)))

            #add batchnorm for each layer except first and last

            generator_model.add(BatchNormalization(momentum=0.9,
                name="gen_bn_"+str(x)))

        generator_model.add(Conv2D(2, kernel_size=3,
            strides=1,
            activation='relu',
            padding='same',
            name="img_output"))

        ab_layers = generator_model(input_fusion)

        composed_image = concatenate([l_image_input, ab_layers], axis=3)

        composed =  Model(inputs=[noise_a_input, noise_b_input, l_image_input], outputs=composed_image)

    composed.summary()
    return composed



def get_generator_optimizer():

    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0
    MOMENTUM=0.9

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    # optim = keras.optimizers.SGD(lr=LR, decay=1e-07, momentum=MOMENTUM)

    return optim


def build_GAN_discriminator():
    with tf.name_scope("discriminator_model"):
        discriminator_model = Sequential()

        D_INPUT_SHAPE = INPUT_SHAPE

        discriminator_model.add(Conv2D(64,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_1",
            input_shape=D_INPUT_SHAPE))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_1"))

        discriminator_model.add(Conv2D(128,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_2"))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_2"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_1"))

        discriminator_model.add(Conv2D(256,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_3"))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_3"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_2"))

        discriminator_model.add(Conv2D(512,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_4"))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_4"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_3"))

        discriminator_model.add(Conv2D(512,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_5"))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_5"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_4"))

        discriminator_model.add(Conv2D(512,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_6"))
        discriminator_model.add(LeakyReLU(0.2, name="discr_leakyReLU_6"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_5"))

        discriminator_model.add(Conv2D(1,
            kernel_size=4,
            strides=2,
            padding='same',
            name="discr_conv2d_7"))
        discriminator_model.add(BatchNormalization(momentum=0.9,name="discr_bn_6"))


        discriminator_model.add(Flatten())
        discriminator_model.add(Dense(1, activation='sigmoid',name="discr_classification"))


        discr_bw_image_input = Input(shape=D_INPUT_SHAPE, name="discr_bw_input")
        generated_image_input = Input(shape=D_INPUT_SHAPE, name="generated_image_input")


        with tf.name_scope("discr_input_mult"):
            model_input = multiply([discr_bw_image_input, generated_image_input])

        classification = discriminator_model(model_input)


        composed =  Model(inputs=[discr_bw_image_input, generated_image_input], outputs=classification, name="discriminator_model")

        optim = get_discriminator_optimizer()

        losses = [ "binary_crossentropy" ]
        composed.compile(optimizer=optim, loss=losses, metrics=[ "accuracy" ])

    return composed

def get_discriminator_optimizer():
    LR = 0.0001
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)
    #optim = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=0.9)

    return optim


def get_gan_optimizer(LR=0.0002, B1=0.5, DECAY=0.01):

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    #optim = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=0.9)
    return optim


