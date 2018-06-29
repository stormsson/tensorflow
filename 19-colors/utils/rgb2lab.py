#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from skimage import io,color
from os import listdir, makedirs
from os.path import isfile, isdir



SCALE_DOWN_CONST = 255.0

def rgb2lab_file(input_file_path):
    rgb = io.imread(input_file_path)
    lab = color.rgb2lab(rgb)
    lab = lab / SCALE_DOWN_CONST


def rgb2labdir(input_directory, output_dir):

    if isdir(output_dir):
        raise(IOError("Directory '%s' already exists. delete it." % output_dir))
    else:
        makedirs(output_dir)


    files = [ f for f in listdir(input_dir)]

    for f in files:
        lab = rgb2lab(input_dir+"/"+f)
        io.imsave(output_dir+"/"+f,lab)


"""
from an rgb image array to a single L channel of LAB format
"""
def extract_L_channel_from_RGB(img):
    img = img / 255.0
    grayscaled_rgb = color.gray2rgb(color.rgb2gray(img))
    # Lab has range -127-128, so we divide by 128 to scale it down in the range -1/1
    lab = color.rgb2lab(grayscaled_rgb) / 128
    return lab[:,:,0]

