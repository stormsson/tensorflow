#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

class User:

    top_categories_length = 3

    def __init__(self, categories):
        self.categories_by_preference = np.random.permutation(range(0, len(categories)))
        self.products_ordered = []
        self.products_ordered_count_by_category = np.zeros(len(categories), dtype=np.int32)

    def getTopCategories(self):
        return self.categories_by_preference[:self.top_categories_length]

    def getProductsOrdered(self):
        return self.products_ordered

    def getProductsOrderedCountByCategory(self):
        return self.products_ordered_count_by_category

    def generateOrder(self, catalog, qty=1):
        for i in range(0, qty):
            tmp = {}
            rnd = np.random.randint(0, 100)
            if rnd < 40:
                category_index = 0
            elif rnd < 60:
                category_index = 1
            elif rnd < 70:
                category_index = 2
            else:
                category_index = np.random.randint(self.top_categories_length, len(self.categories_by_preference) - self.top_categories_length)

            category_id = self.categories_by_preference[category_index]
            self.products_ordered.append(catalog.pickProducts(category_id))
            self.products_ordered_count_by_category[category_id] += 1






