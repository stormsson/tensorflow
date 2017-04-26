#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn
import math
import numpy
import random
import os
import csv
from utils import Standardizer
from utils import StandardizerContext
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
basePath = os.path.dirname(os.path.realpath(__file__))
saveFilePath = "./model.tflearn"


# RowNumber,CustomerId,Surname,CreditScore,FRANCE,GERMANY,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited
# REMOVED  ,   REMOVED,REMOVED,CreditScore,FRANCE,GERMANY,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary => Exited
# Gender: 0 = Female, 1 = Male



def generateData():
    data = {'in':[], 'out':[]}

    with open('data_elab.csv', 'rb') as csvFile:
        reader = csv.reader(csvFile, delimiter=",")
        for row in reader:
            data['in'].append(numpy.array(row[:-1], dtype=numpy.float64))
            data['out'].append(numpy.array(row[-1:], dtype=numpy.float64))

    return data

def generateGraph():
    g = tflearn.input_data(shape=[None, 11])

    # suggerimento: usare la media della somma di #input + #output
    # (11 + 1) / 2 => 6
    g = tflearn.fully_connected(g, 32, activation='relu')
    # g = tflearn.dropout(g, .9)
    g = tflearn.fully_connected(g, 32, activation='relu')
    # g = tflearn.dropout(g, .8)
    g = tflearn.fully_connected(g, 2, activation='sigmoid')

    return g


data = generateData()
cx = StandardizerContext(data['in'], Standardizer.NORMALIZATION_TYPE_MIN_DIFF )
# cx = StandardizerContext(data['in'], Standardizer.NORMALIZATION_TYPE_MEAN_ABSzMAX )
data['in'] = cx.getData()
pippo = []
for e in data['out']:
    if(e == 1):
        pippo.append([1.0, 0.0])
    else:
        pippo.append([0.0, 1.0])

data['out'] = pippo


with tf.Graph().as_default():
    g = generateGraph()

    sgd = tflearn.optimizers.SGD(learning_rate=.001,
        lr_decay=.98)
    adam = tflearn.optimizers.Adam(
        learning_rate=.001
    )
    rmsprop = tflearn.optimizers.RMSProp(
        learning_rate=.001,
        decay=.99
    )

    # losses: softmax_categorical_crossentropy, categorical_crossentropy, binary_crossentropy, mean_square, hinge_loss, roc_auc_score, weak_cross_entropy_2d
    g = tflearn.regression(g, optimizer=adam,
        batch_size=64,
        loss='categorical_crossentropy'
        # loss='binary_crossentropy'
        # loss='mean_square'
        )

    m = tflearn.DNN(g,
        tensorboard_verbose=3,
        tensorboard_dir="./tensorboard",
        # checkpoint_path="./checkpoints/",
        # best_checkpoint_path="./checkpoints/best/",
        max_checkpoints=0
    )

    if  os.path.exists(saveFilePath+".index"):
        m.load(saveFilePath)
        print("Restored model weights from ", saveFilePath)

    else:
        print("CREATING")
        # print(data['in'][0], data['out'][0])
        # exit()
        m.fit(data['in'], data['out'], n_epoch=100, shuffle=True,  snapshot_epoch=False, show_metric=True, validation_set=0.20, run_id="bank_classification")
        # m.save(saveFilePath)


print("Output categories meaning: [ LEAVING, STAYING ]")

errors = 0
num_elem = 500

correct_elements = []
predictions = []
for x in xrange(0, num_elem):
    data_in = data['in'][x]

    out = data['out'][x]
    correct_elements.append("Leaving" if out[0] else "Staying")

    prediction = m.predict([data_in])

    if(prediction[0][0] > prediction[0][1]):
        print("Prediction: 1, confidence: ", prediction[0][0], "expected: ", out," ",prediction)
        predictions.append("Leaving")
        if out[0] == 0:
            errors = errors+1
    else:
        print("Prediction: 0 , confidence: ", prediction[0][1], "expected: ", out," ",prediction)
        predictions.append("Staying")
        if out[1] == 0:
            errors = errors+1


print("Erros: %d / %d\n" % (errors, num_elem))

cm = confusion_matrix(correct_elements, predictions, labels=["Leaving", "Staying"])
print("\tLeaving\tStaying")
print("Leaving\t%d\t%d" % (cm[0][0],cm[0][1]))
print("Staying\t%d\t%d\n" % (cm[1][0],cm[1][1]))

print("False positives: ",cm[0][1])
print("False negatives: ",cm[1][0])




