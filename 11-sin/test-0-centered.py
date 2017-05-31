from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn
import math
import random
import os

import matplotlib.pyplot as plt

basePath = os.path.dirname(os.path.realpath(__file__))
saveFilePath = "./model.tflearn"

pi = math.pi

X = []
Y = []

def generateData(qty=1000):
    data = {
        'in': [],
        'out': []
    }
    for el in xrange(1, qty):
        rnd = 2 * random.random() -1
        #in: rads
        data['in'].append([rnd])
        X.append(rnd)

        data['out'].append([math.sin(pi * rnd)])
        Y.append(math.sin(pi * rnd))

    return data

data = generateData(2000)

plt.plot(X,Y, 'g^')
# plt.show()
# exit()

saver = tf.train.Saver(max_to_keep=5)

with tf.Graph().as_default():
    # graph definition
    g = tflearn.input_data(shape=[None, 1])

    g = tflearn.fully_connected(g, 32, activation='relu')
    g = tflearn.fully_connected(g, 32, activation='relu')

    g = tflearn.fully_connected(g, 1, activation='linear')
    sgd = tflearn.optimizers.SGD(learning_rate=.1,
        lr_decay=.98)

    adam = tflearn.optimizers.Adam(
        learning_rate=0.01
    )
    g = tflearn.regression(g, optimizer=sgd,
         # loss='categorical_crossentropy')
        # loss='binary_crossentropy')
        loss='mean_square')

    m = tflearn.DNN(g)

    if False and os.path.exists(saveFilePath+".index"):
        # m.load(saveFilePath)
        print("Restored model weights from ", saveFilePath)

    else:
        print("CREATING")
        m.fit(data['in'], data['out'], n_epoch=100, shuffle=True,  snapshot_epoch=False, show_metric=True, validation_set=0.10)
        # m.save(saveFilePath)


    # testing
    rX = []
    rY = []
    for x in xrange(1,100):
        rnd = random.random() * 2 -1
        rX.append(rnd)
        res = m.predict([[rnd]])
        rY.append( res[0][0] )


    plt.plot(rX, rY, 'ro')

    plt.show()