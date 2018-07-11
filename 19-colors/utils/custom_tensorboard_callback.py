#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from keras.callbacks import TensorBoard
import keras.backend as K


class CustomTensorBoard(TensorBoard):


    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 starting_epoch=1):

        self.starting_epoch = starting_epoch
        self.epoch_counter = self.starting_epoch

        super(CustomTensorBoard, self).__init__(log_dir,
                 histogram_freq,
                 batch_size,
                 write_graph,
                 write_grads,
                 write_images,
                 embeddings_freq,
                 embeddings_layer_names,
                 embeddings_metadata,
                 )



    def on_epoch_end(self, epoch, logs=None):
        super(CustomTensorBoard, self).on_epoch_end(self.epoch_counter, logs)
        self.epoch_counter += 1



    #     pippo = tf.Summary()
    #     pippo_value = pippo.value.add()
    #     pippo_value.simple_value = 5
    #     pippo_value.tag = "GESUCRISTO"

    #     self.writer.add_summary(pippo, epoch)

    #     super(CustomTensorBoard, self).on_epoch_end(epoch, logs)

    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)