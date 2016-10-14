#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import tensorflow as tf


basePath = os.path.dirname(os.path.realpath(__file__))
saveFilePath = basePath +"/saveload.ckpt"

state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# esempio di eval invece di run
# hot.eval(session=sess)

# ops to save and load
saver = tf.train.Saver()
with tf.Session() as sess:
    if not os.path.exists(saveFilePath):
        print "state file not found"
        sess.run(init_op)
    else:
        print "state file found"
        saver.restore(sess, saveFilePath)

    # Print the initial value of 'state'
    print("Initial state: %s " % (sess.run(state)))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

    save_path = saver.save(sess, saveFilePath)
    sess.close()