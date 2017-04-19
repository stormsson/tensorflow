#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy

class Standardizer:

    @staticmethod
    def normalize(values):
        mean = numpy.mean(values)
        standardized = (values - numpy.mean(values))
        absmax = abs(max(standardized.min(), standardized.max(), key=abs))

        if(absmax):
            standardized = standardized / absmax

        return standardized, mean, absmax

    @staticmethod
    def denormalize(standardized_values, mean, absmax):
        return [x * absmax + mean for x in standardized_values]
