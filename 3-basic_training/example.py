#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf
import numpy as np

# see https://www.youtube.com/watch?v=bvHgESVuS6Q
x = tf.placeholder("float")
y = tf.placeholder("float")


# --- inizio definizione grafico modello
# w[0] e' la variabile "a" nell'equazione , w[1] e' "b"
w = tf.Variable([1., 2.], name="w")

# model: y = a*x +b
y_model = tf.mul(x, w[0]) + w[1]

# misura dell'errore. il nostro obiettivo e' minimizzarlo.
error = tf.square(y - y_model)

train_operation = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# --- fine definizione grafico modello
init = tf.initialize_all_variables()

# valori "segreti" , sono i risultati a cui il modello deve tendere
a = 2
b = 6

with tf.Session() as session:
    session.run(init)
    for i in range(10000):
        x_value = np.random.rand()
        y_value = x_value * a + b
        session.run(train_operation, feed_dict = {x: x_value, y: y_value})
    w_value = session.run(w)

    print ("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
