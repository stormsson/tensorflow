#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from skimage import io,color
from os import listdir, makedirs
from os.path import isfile, isdir


SCALE_DOWN_CONST = 255.0

def rgb2lab(input_file_path):
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