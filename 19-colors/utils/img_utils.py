#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.color import rgb2gray as sk_rgb2gray

from os.path import isfile, isdir

import numpy as np

def rgb2gray(img):
    return gray2rgb(sk_rgb2gray(img / 255.0))



class color_manager(object):

    def __init__(self):
        self.image_path = None
        self.img_array = None
        self.scale_factor = 1.0
        self.image_type = ""

    def loadRGB(self, image_path):
        self.image_path = image_path
        self.img_array = img_to_array(load_img(image_path))
        self.scale_factor  = 255.0
        self.image_type="RGB"

        return self

    def fromRGBArray(self, data):
        self.img_array = data
        self.scale_factor  = 255.0
        self.image_type="RGB"

        return self


    def RGBtoLAB(self):
        self.img_array = rgb2lab(self.img_array)

        self.scale_factor = 128.0
        self.image_type="LAB"

        return self

    def LABtoRGB(self):
        self.img_array = lab2rgb(self.img_array)
        self.scale_factor = 255.0
        self.image_type = "RGB"

        return self

    def toGrayScaleRGB(self):
        self.img_array = gray2rgb(rgb2gray(self.img_array))
        self.image_type="GRAY"

        return self

    def scaleDown(self):
        self.img_array = self.img_array / self.scale_factor

        return self

    def scaleUp(self):
        self.img_array = self.img_array * self.scale_factor

        return self


    def getChannel(self, channel):
        if channel<0 or channel > 2:
            raise ValueError("chanel must be 0,1,2")

        return self.img_array[:, :, channel]

    def getImageArray(self):
        return self.img_array

    def getIntArray(self):
        return self.img_array.astype(np.uint8)
