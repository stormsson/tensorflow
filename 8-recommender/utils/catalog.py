#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import generator
import numpy as np

import pprint

pp = pprint.PrettyPrinter(indent=4)

class Catalog:
    def __init__(self, num_categories, num_products ):
        self.categories = generator.generateCategories(num_categories)
        products = generator.generateProducts(num_categories, num_products)

        self.products_by_category  = []
        for i in range(0, num_categories):
            self.products_by_category.append([])

        for p in products:
            cat = p['main_category']
            self.products_by_category[cat].append(p)

        # for index, c in enumerate(self.products_by_category):
        #     print "category: %d, len: %d" % (index, len(c))


    def getCategories(self):
        return self.categories

    def getProductsByCategory(self, category=None):
        if not category:
            return self.products_by_category

        return self.products_by_category[category]

    def pickProducts(self, category, qty=1):
        result = []

        for x in xrange(0, qty):
            print category
            cat = self.products_by_category[category]
            rnd = np.random.randint(0, len(cat))
            result.append(cat[rnd])
        return result


    def categoriesToWeights(self, weights_categories):
        cat_length = len(self.categories)
        result = np.zeros(cat_length, dtype=np.float32)
        for i in range(0, len(weights_categories)):
            if weights_categories[i] > cat_length-1:
                raise IndexError("Category %d not in catalog " % weights_categories[i])
            result[weights_categories[i]] = 1.0

        return result
