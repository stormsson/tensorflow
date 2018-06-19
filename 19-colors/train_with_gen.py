#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import keras
import math
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image
import random

from bcolors import bcolors
import sys

from generator import createTrainGenerator
from model import buildModel, getInception


IMG_HEIGHT = 255
IMG_WIDTH = 255
IMG_CHANNELS = 3

input_shape =  ( IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS )


inception = getInception()

print(inception)


#train_generator = createTrainGenerator(images, masks)

#model = buildmodel(input_shape)


#model.fit_generator(
#    train_generator,
#    steps_per_epoch=2000,
#    epochs=50)

