#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from utils import generator
from utils.user import User
from utils.catalog import Catalog

import numpy as np

import pickle

np.random.seed(100)

c = Catalog(10, 5000)

# with open(pickle_file, 'rb') as f:
#             dataset = pickle.load(f)


orders = []
training_categories = []
for _ in range(1000):
    userInstance = User(c.getCategories())
    userInstance.generateOrder(c, 50)



userInstance = User(c.getCategories())
# print userInstance.getTopCategories()

print userInstance.getTopCategories()
print c.categoriesToWeights(userInstance.getTopCategories())

print userInstance.getProductsOrderedCountByCategory()

print c.categoriesToWeights([0,1,4,6])
print c.categoriesToWeights([0,1])
print c.categoriesToWeights([0,1])


n = Network("")