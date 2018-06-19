#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from skimage import io,color
from os import listdir, makedirs
from os.path import isfile, isdir

from utils.rgb2lab import rgb2labdir

import sys

if len(sys.argv) != 3:
	print("use rgb2lab inputdir outputdir")
	exit()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

rgb2labdir(input_dir, output_dir)

