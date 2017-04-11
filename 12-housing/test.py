from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn
import math
import numpy
import random
import os
from sklearn.preprocessing import normalize


import matplotlib.pyplot as plt

basePath = os.path.dirname(os.path.realpath(__file__))
saveFilePath = "./model.tflearn"


HIDDEN_MQ_PRICE = 1000
HIDDEN_GARDEN_MQ_PRICE = 100
HIDDEN_PARK_PRICE = 10000


def generateData(qty=1000):
    data = {
        'in': [],
        'out': []
    }

    mqs = []
    mqgs = []
    parkss = []
    prices = []
    bias = []

    for el in xrange(1, qty):
        mq = 50 + random.randint(0, 150)
        mqg = 50 + random.randint(0, 200) * 10
        parks = random.randint(0, 2)
        price = mq * HIDDEN_MQ_PRICE + mqg * HIDDEN_GARDEN_MQ_PRICE + parks * HIDDEN_PARK_PRICE
        price = mq * HIDDEN_MQ_PRICE


        mqs.append( mq )
        mqgs.append( mqg )
        parkss.append( parks )
        prices.append( price )
        bias.append( 1.0 )

    # mqs_stats = {
    #     'mean': numpy.mean(mqs, dtype=numpy.float64),
    #     'stddev': numpy.std(mqs, dtype=numpy.float64)
    # }
    mqs_norm = normalize(numpy.array(mqs).reshape(1, -1), norm='max')[0]

    # mqgs_stats = {
    #     'mean': numpy.mean(mqgs, dtype=numpy.float64),
    #     'stddev': numpy.std(mqgs, dtype=numpy.float64)
    # }
    mqgs_norm = normalize(numpy.array(mqgs).reshape(1, -1), norm='max')[0]

    # parkss_stats = {
    #     'mean': numpy.mean(parkss, dtype=numpy.float64),
    #     'stddev': numpy.std(parkss, dtype=numpy.float64)
    # }
    parkss_norm = normalize(numpy.array(parkss).reshape(1, -1), norm='max')[0]

    # prices_stats = {
    #     'mean': numpy.mean(prices, dtype=numpy.float64),
    #     'stddev': numpy.std(prices, dtype=numpy.float64)
    # }
    prices_norm = normalize(numpy.array(prices).reshape(1,-1), norm='max')[0]




    # print(mqs, "\n", mqs_stats, "\n", mqs_norm)
    # print(mqgs, "\n", mqgs_stats, "\n", mqgs_norm)
    # print(parkss, "\n", parkss_stats, "\n", parkss_norm)
    # print(prices, "\n", prices_stats, "\n", prices_norm)



    data['in'] = [list(a) for a in zip(mqs_norm, mqgs_norm, parkss_norm, bias)]
    data['in'] = [list(a) for a in zip(mqs)]

    data['original'] = zip(mqs, mqgs, parkss)

    data['max_price'] = max(prices)
    for o in prices:
        #aggiungo valori normalizzati a max
        data['out'].append([o/data['max_price']])


    # print(data)
    # exit()
    return data

def generateGraph():
    g = tflearn.input_data(shape=[None, 1])

    g = tflearn.fully_connected(g, 128, activation='sigmoid')
    g = tflearn.fully_connected(g, 128, activation='relu')



    g = tflearn.fully_connected(g, 1, activation='linear')

    return g


data = generateData(2000)



with tf.Graph().as_default():
    # graph definition

    g = generateGraph()

    sgd = tflearn.optimizers.SGD(learning_rate=.07,
        lr_decay=.98)
    adam = tflearn.optimizers.Adam(
        learning_rate=0.01
    )
    g = tflearn.regression(g, optimizer=sgd,
        batch_size=64,
         # loss='categorical_crossentropy')
        # loss='binary_crossentropy')
        loss='mean_square')

    m = tflearn.DNN(g, tensorboard_verbose=3, tensorboard_dir="./tensorboard")

    if False and os.path.exists(saveFilePath+".index"):
        # m.load(saveFilePath)
        print("Restored model weights from ", saveFilePath)

    else:
        print("CREATING")
        m.fit(data['in'], data['out'], n_epoch=180, shuffle=True,  snapshot_epoch=False, show_metric=True, validation_set=0.10)
        # m.save(saveFilePath)


    # testing
    rX = []
    rY = []
    for x in xrange(1,100):
        mq =random.randint(50, data['max_price']/1000)
        mqg = 50 + random.randint(0, 200) * 10
        parks = random.randint(0, 2)
        record = [ mq, mqg, parks ]

        expected_price = mq * HIDDEN_MQ_PRICE + mqg * HIDDEN_GARDEN_MQ_PRICE + parks * HIDDEN_PARK_PRICE
        expected_price = mq * HIDDEN_MQ_PRICE

        # res = m.predict([record])
        res = m.predict([[mq]])
        error = expected_price  - res[0][0] * data['max_price']
        if( error > -1 and error < 1):
            error = 0

        print("predicted: ",res[0][0] *data['max_price'], "expected: ",expected_price,"error: ",error)
        rX.append(x)
        rY.append( error )
    print("max price:",data['max_price'])
    print("weight",m.get_weights(g.W))


    plt.plot(rX, rY, 'ro')
    plt.show()