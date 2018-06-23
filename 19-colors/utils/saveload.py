#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import load_model as keras_load_model
from os.path import isfile
from bcolors import bcolors

def restore_model(restore_file):

    if isfile(restore_file):
        model = keras_load_model(restore_file)
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


def save_model(model, model_name, directory_path=False):

    if not directory_path:
        directory_path = "models/"

    if directory_path[-1] != "/":
        directory_path = directory_path+ "/"

    model_config_path = directory_path+model_name+"_config.txt"
    with open(model_config_path, "w") as f:
        print(bcolors.OKGREEN+"Writing model config to: "+model_config_path+"!"+bcolors.ENDC)

        f.write(str(model.get_config()))

    try:
        model.save("models/"+model_name)
        print(bcolors.OKGREEN+"Model saved to: models/"+model_name+"!"+bcolors.ENDC)

    except IOError as e:
        model.save(model_name)