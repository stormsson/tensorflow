#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


basePath = os.path.dirname(os.path.realpath(__file__))

filename = basePath+"/emma.jpg"
raw_image_data = mpimg.imread(filename)

x = tf.Variable(raw_image_data, name='x')
model = tf.initialize_all_variables()

# https://www.tensorflow.org/versions/r0.10/api_docs/python/array_ops.html#transpose
transpose_operation = tf.transpose(x, perm=[1,0,2])

with tf.Session() as session:
    session.run(model)
    result = session.run(transpose_operation)

plt.imshow(result)
plt.show()

