#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


basePath = os.path.dirname(os.path.realpath(__file__))
filename = basePath+"/emma.jpg"

def example1():
    image = tf.placeholder("uint8", [ None, None, 3 ])

    # https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#transpose
    transpose_operation = tf.transpose(image, perm=[1,0,2])

    with tf.Session() as session:

        result = session.run(transpose_operation, feed_dict={image: mpimg.imread(filename)})

    plt.imshow(result)

example1()