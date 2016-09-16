#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import tensorflow as tf
import tflearn

from tflearn.data_utils import load_csv


basePath = os.path.dirname(os.path.realpath(__file__))


def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]

    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.

    return np.array(data, dtype=np.float32)



from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# esempio:
#  sibsp = siblings/spouses aboard
#  parch = parents/children aboard
#
# survived    pclass               name                sex    age  sibsp   parch   ticket      fare
#    1          1     Aubart, Mme. Leontine Pauline   female   24    0      0     PC 17477    69.3000


# Load CSV file, indicate that the first column represents labels
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

columns_to_ignore = [1, 6]
preprocess(data, columns_to_ignore )

# input data = 6 feature, dimensioni a None significa sconosciute, quindi viene cambiato a runtim

net = tflearn.input_data(shape=[None, 6])
# 2 strati fully connected, 32 neuroni
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)

# un terzo strato di output, 2 neuroni (vogliamo sapere solo sopravvissuto si/no)
net = tflearn.fully_connected(net, 2, activation='softmax')

net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], columns_to_ignore)

# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])