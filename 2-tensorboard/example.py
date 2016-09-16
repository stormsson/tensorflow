#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf


basePath = os.path.dirname(os.path.realpath(__file__))

var = tf.Variable(tf.random_normal([50], seed=1324), name="variabile")


with tf.name_scope('summaries'):
    tf.histogram_summary("istogramma della normale", var)
    tf.scalar_summary('max', tf.reduce_max(var))
    tf.scalar_summary('min', tf.reduce_min(var))

    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.scalar_summary('stddev', mean)

    merged_summaries = tf.merge_all_summaries()


init_op = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.train.SummaryWriter(basePath + '/log', sess.graph)
    print(sess.run(var))
    summary = sess.run(merged_summaries)
    writer.add_summary(summary, 1)
