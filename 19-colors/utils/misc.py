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
def generate_noise(amount=1, size=256, channels=3, noise_range=(0, 1) ):
    noise = np.random.uniform(noise_range[0], noise_range[1], (amount, size, size, channels))
    return noise
