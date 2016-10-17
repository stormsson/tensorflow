#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

def generateCategories(qty=10):
    result = []
    for i in range (0, qty):
        result.append("Category %d" % i)

    return result

def generateProducts(num_categories, qty=10):
    result = []

    for i in range (0, qty):
        tmp = {}
        tmp['name'] = "Product %s" % i
        tmp['categories'] = np.random.permutation(range(0, num_categories))
        tmp['main_category'] = tmp['categories'][0]
        tmp['score_per_category'] = np.zeros(num_categories)
        # distr = np.random.normal(size=10, loc=.5, scale= .2)
        distr = np.random.noncentral_chisquare(df=2, nonc=.5, size=10)
        distr = sorted(distr/max(distr))
        # distr = sorted(distr)
        distr.reverse()
        tmp['score_per_category'] = distr
        result.append(tmp)

    return result
