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
