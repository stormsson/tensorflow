#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn
import math
import numpy
import random
import os
from utils import Standardizer


import matplotlib.pyplot as plt

basePath = os.path.dirname(os.path.realpath(__file__))
saveFilePath = "./model.tflearn"


HIDDEN_MQ_PRICE = 1000
HIDDEN_GARDEN_MQ_PRICE = 100
HIDDEN_PARK_PRICE = 10000

# x = [ 179, 187, 67, 96, 150]
# print("orig: ", x)
# asd = Standardizer.normalize(x)
# print("norm: ", asd)
# print("avgdiff: ", x - asd[1])
# print ("denorm:", Standardizer.denormalize(asd[0],asd[1],asd[2]))
# exit()

def doNormalization(mqs, mqgs, parkss, prices):

    normalized_mqs = {}
    normalized_mqgs = {}
    normalized_parks = {}
    normalized_prices = {}


    normalized_mqs['values'], normalized_mqs['mean'], normalized_mqs['absmax'] = Standardizer.normalize(mqs)
    normalized_mqgs['values'], normalized_mqgs['mean'], normalized_mqgs['absmax'] = Standardizer.normalize(mqgs)
    normalized_parks['values'], normalized_parks['mean'], normalized_parks['absmax'] = Standardizer.normalize(parkss)
    normalized_prices['values'], normalized_prices['mean'], normalized_prices['absmax'] = Standardizer.normalize(prices)

    return normalized_mqs, normalized_mqgs, normalized_parks, normalized_prices


def generateData(qty=1000):
    data = {
        'in': [],
        'out': []
    }

    mqs = []
    mqgs = []
    parkss = []
    prices = []

    seed = random.randint(0, 1000000)
    seed = 410374
    print("SEED: %d" % seed)
    random.seed(seed)

    for el in xrange(0, qty):
        mq = 50 + random.randint(0, 250)
        mqg = 50 + random.randint(0, 100) * 10
        parks = random.randint(0, 2)

        # only mq prices
        price = mq * HIDDEN_MQ_PRICE

        # mq and mqg prices
        price += mqg * HIDDEN_GARDEN_MQ_PRICE

        # mq and mqg and parks prices
        price += parks * HIDDEN_PARK_PRICE


        mqs.append( mq )
        mqgs.append( mqg )
        parkss.append( parks )
        prices.append( price )

    features = {
        'mq': {},
        'mqg': {},
        'parks': {},
        'price': {}
    }

    features['mq'], features['mqg'],features['parks'], features['price'] = doNormalization(mqs, mqgs, parkss, prices)
    # print(mqs)
    # print(features['mq']['values'])
    # print("--")
    # print(prices)
    # print(features['price']['values'])

    # fee = zip(features['mq']['values'], features['price']['values'])
    # for x in fee:
    #     plt.plot(x[0], x[1], 'ro')

    # plt.show()
    # exit()


    # NORMALIZED DATA AS INPUT SECTION #

    # only use mqs as input
    data['in'] = [list(a) for a in zip(
        features['mq']['values'],
        features['mqg']['values'],
        features['parks']['values']
        )]

    data['original'] = zip(mqs, mqgs, parkss)

    for o in features['price']['values']:
        #aggiungo valori normalizzati a max
        data['out'].append([o])


    # print(features)
    # exit()
    return data, features

def generateGraph():
    g = tflearn.input_data(shape=[None, 3])
    g = tflearn.fully_connected(g, 16, activation='relu')
    g = tflearn.fully_connected(g, 16, activation='relu')
    g = tflearn.fully_connected(g, 16, activation='relu')


    g = tflearn.fully_connected(g, 1, activation='linear')


    # g = tflearn.input_data(shape=[None, 2])

    # g = tflearn.fully_connected(g, 128, activation='relu')

    # g = tflearn.fully_connected(g, 64, activation='relu')
    # g = tflearn.fully_connected(g, 64, activation='relu')
    # g = tflearn.fully_connected(g, 1, activation='linear')

    return g

data, features = generateData(8000)

import seaborn as sns
import pandas as pd

# sns.set(style="darkgrid")

sns.set(color_codes=True)

data = [list(a) for a in zip(features['mq']['values'], features['price']['values'])]

# kind={ “scatter” | “reg” | “resid” | “kde” | “hex” }, optional

sns.jointplot(x=features['mq']['values'], y=features['price']['values'], kind='hex')
plt.show()
exit()


PARAM_LEARNING_RATE = .001
PARAM_LEARNING_RATE_DECAY = .95
PARAM_NUM_EPOCH = 20

with tf.Graph().as_default():
    # graph definition

    g = generateGraph()

    sgd = tflearn.optimizers.SGD(learning_rate=PARAM_LEARNING_RATE,
        lr_decay=PARAM_LEARNING_RATE_DECAY)
    adam = tflearn.optimizers.Adam(
        learning_rate=PARAM_LEARNING_RATE
    )
    g = tflearn.regression(g, optimizer=adam,
        batch_size=64,
         # loss='categorical_crossentropy')
        # loss='binary_crossentropy')
        loss='mean_square')

    m = tflearn.DNN(g,
        tensorboard_verbose=3,
        tensorboard_dir="./tensorboard",
        # checkpoint_path="./checkpoints/",
        # best_checkpoint_path="./checkpoints/best/",
        # max_checkpoints=0
    )

    if False and os.path.exists(saveFilePath+".index"):
        m.load(saveFilePath)
        print("Restored model weights from ", saveFilePath)

    else:
        print("CREATING")
        m.fit(data['in'], data['out'], n_epoch=PARAM_NUM_EPOCH, shuffle=True,  snapshot_epoch=False, show_metric=True, validation_set=0.10)
        # m.save(saveFilePath)


    # testing
    rX = []
    rY = []



    for x in xrange(1,200):
        mq =random.randint(50, 250)
        mqg = 50 + random.randint(0, 100) * 10
        parks = random.randint(0, 2)
        record = [ mq, mqg, parks ]

        expected_price = mq * HIDDEN_MQ_PRICE
        expected_price += mqg * HIDDEN_GARDEN_MQ_PRICE
        expected_price += parks * HIDDEN_PARK_PRICE

        mq = (mq - features['mq']['mean'] ) / features['mq']['absmax']
        mqg = (mqg - features['mqg']['mean'] ) / features['mqg']['absmax']
        parks = (parks - features['parks']['mean'] ) / features['parks']['absmax']

        res = m.predict([[mq, mqg, parks]])
        converted_price = Standardizer.denormalize(
                res[0],
                features['price']['mean'],
                features['price']['absmax']
            )
        error = expected_price  - converted_price[0]

        print(
            "predicted: ",converted_price[0],
            "expected: ",expected_price,
            "error: ",error
            )
        rX.append(x)
        rY.append( error )
    max_price = features['price']['absmax'] + features['price']['mean']
    print("max price:",max_price)
    print("avg error: ", numpy.mean(rY))
    print("std dev: ", numpy.std(rY))

    one_percent = abs(max_price / 100)
    less_than_one_percent = [ a for a in rY if abs(a) < one_percent ]
    less_than_half_percent = [ a for a in rY if abs(a) < one_percent/2 ]
    print("errors < 1% : ", len(less_than_one_percent),"/200 -> ",len(less_than_one_percent)/2,"%")
    print("errors < 0.5% : ", len(less_than_half_percent),"/200 -> ",len(less_than_half_percent)/2,"%")


    plt.plot(rX, rY, 'ro')
    plt.title('Prediction errors')
    plt.xlabel('prediction #')
    plt.ylabel('error')




    plt.show()