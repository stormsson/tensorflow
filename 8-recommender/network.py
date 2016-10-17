#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf
class Network():
    basePath = os.path.dirname(os.path.realpath(__file__))
    dataPath = basePath + "../data/8-recommender/"

    def __init__(self, name):
        self.saver = tf.train.Saver()
        self.name = name
        self.saveFilePath = dataPath+name+".ckpt"



    def setup(self, catalog):
        num_categories = len(catalog.getCategories())

        # ricevero' un numero non definito di righe, contenenti num_categories
        inputs = tf.placeholder(tf.float32, [None, num_categories])

        weights = tf.Variable(tf.zeros([num_categories, num_categories]))
        bias = tf.Variable(tf.zeros(num_categories))

        linear_regression = tf.nn.softmax(tf.matmul(inputs, weights) + bias)

        correct_categories = tf.placeholder(tf.float32, [None, num_categories])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_categories * tf.log(linear_regression), reduction_indices=[1]))

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.init_op = tf.initialize_all_variables()



    def train(self, iterations, batch_input, expected_output):
        with tf.Session() as sess:
            if not os.path.exists(self.saveFilePath):
                print "state file not found"
                sess.run(self.init_op)
            else:
                print "state file found"
                saver.restore(sess, self.saveFilePath)

            for i in range(iterations):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(self.train_step, feed_dict={linear_regression: batch_xs, correct_categories: batch_ys})