#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf
import numpy as np


class test():
    def __init__(self, name):
        self.name = name

    def setup(self):
        self.state = tf.Variable(0, name="counter")
        one = tf.constant(1)
        new_value = tf.add(self.state, one)
        self.update = tf.assign(self.state, new_value)

        self.init_op = tf.initialize_all_variables()



    def run(self):

        # init_op, state, update  = self.setup()
        with tf.Session() as sess:
          # Run the 'init' op
          sess.run(self.init_op)
          # Print the initial value of 'state'
          print(sess.run(self.state))
          # Run the op that updates 'state' and print 'state'.
          for _ in range(3):
            sess.run(self.update)
            print(sess.run(self.state))


x = test("asdfg")
x.setup()
x.run()
