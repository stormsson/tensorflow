#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import LeakyReLU

import keras.backend as K

import tensorflow as tf

from bcolors import bcolors
from os.path import isfile


def buildModel(input_shape):


    # img_rows , img_cols, channels

    # r_eye X, r_eye Y , l_eye X, l_eye Y
    output_size = 4

    with tf.device('/gpu:0'):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(11, 11),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(64, (5, 5),
            activation='relu',
            padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Conv2D(64, (3, 3),
            activation='relu',
            padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(48, (3, 3),
            activation='relu',
            padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))


        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))


        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))


        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))



        model.add(Dense(output_size, activation='sigmoid'))

    return model


def restoreModel(restore_file):

    if isfile(restore_file):
        model = load_model(restore_file)
        print(bcolors.OKGREEN+"restored model "+restore_file+"!"+bcolors.ENDC)

        print("count: ",len(model.layers))
        num_layers = len(model.layers)
        tail_layers_to_train = False

        if tail_layers_to_train:
            for layer in model.layers[:num_layers -tail_layers_to_train]:
                layer.trainable = False

            for layer in model.layers[num_layers -tail_layers_to_train:]:
                layer.trainable = True


        for i,layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)


        return model


def saveModel(model, model_name, directory_path=False):

    if not directory_path:
        directory_path = "models/"

    model_config_path = directory_path+model_name+"_config.txt"
    with open(model_config_path, "w") as f:
        print(bcolors.OKGREEN+"Writing model config to: "+model_config_path+"!"+bcolors.ENDC)

        f.write(str(model.get_config()))

    try:
        model.save("models/"+model_name)
        print(bcolors.OKGREEN+"Model saved to: models/"+model_name+"!"+bcolors.ENDC)

    except IOError as e:
        model.save(model_name)


from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf

def getInception():
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    inception.graph = tf.get_default_graph()
    return inception


def build_GAN_generator(depth=10):
    model = Sequential()

    # input = 256x256 images, 3 channels (LAB, not RGB)
    model.add(Input(shape=(256, 256, 3, )))

    # [depth] Conv2D Layers
    for x in xrange(0, depth):
        model.add(Conv2D(64, (3, 3),
            activation='relu',
            padding='valid'))

        #add batchnorm for each layer except first and last
        if (x>0) and ( x < depth-1):
            model.add(BatchNormalization(momentum=0.9))

    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    model.compile(optimizer=optim, loss='mse')

    return model

def build_GAN_discriminator():
    model = Sequential()

    model.add(Input(shape=(256, 256, 3, )))

    model.add(Conv2D(64, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))

    model.add(Conv2D(128, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(256, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(512, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Conv2D(1, (4, 4),
        strides=(2, 2),
        activation=LeakyReLU(0.2),
        padding='valid'))
    model.add(BatchNormalization(momentum=0.9))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))


    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    model.compile(optimizer=optim, loss='mse')

    return model

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def l2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)



