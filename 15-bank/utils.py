#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy

class Standardizer:
    NORMALIZATION_TYPE_MEAN_ABSMAX = "mean_absmax"
    NORMALIZATION_TYPE_MIN_DIFF = "min_diff"

    @staticmethod
    def normalize(values):
        mean = numpy.mean(values)
        standardized = (values - numpy.mean(values))
        absmax = abs(max(standardized.min(), standardized.max(), key=abs))

        if(absmax):
            standardized = standardized / absmax

        standardized = numpy.float64(standardized)
        mean = numpy.float64(mean)
        absmax = numpy.float64(absmax)

        norm_info = {
            "type": Standardizer.NORMALIZATION_TYPE_MEAN_ABSMAX,
            "mean": mean,
            "absmax": absmax
        }
        return standardized, norm_info

    @staticmethod
    def denormalize(standardized_values, mean, absmax):
        return [x * absmax + mean for x in standardized_values]



    @staticmethod
    def normalizeMinDiff(values):
        min_value = min(values)
        max_value = max(values)
        standardized = numpy.array([ el - min_value for el in values ], dtype=numpy.float64)

        diff = max_value - min_value

        if(diff):
            standardized = standardized / diff


        norm_info = {
            "type": Standardizer.NORMALIZATION_TYPE_MIN_DIFF,
            "min": min_value,
            "diff": diff
        }

        return standardized, norm_info

    @staticmethod
    def denormalizeMinDiff(standardized_values, min_value , diff):
        return [ x * diff + min_value for x in standardized_values ]

# ascolto FF7 - Prelude
class StandardizerContext:
    def __init__(self, data, normalization_type=Standardizer.NORMALIZATION_TYPE_MEAN_ABSMAX):
        self.data = data
        self.normalizations = {}
        self.normalization_type = normalization_type

        self.normalizeColumns()

    def getData(self):
        return self.data

    def getNormalizations(self):
        return self.normalizations

    def normalizeColumns(self):


        for column_index in xrange(0, len(self.data[0])):
            to_normalize  = [ numpy.float64(row[column_index]) for row in self.data ]

            # da migliorare
            if(self.normalization_type == Standardizer.NORMALIZATION_TYPE_MEAN_ABSMAX):
                normalized_column, normalization_info = getattr(Standardizer, "normalize")(to_normalize)
            else:
                normalized_column, normalization_info = getattr(Standardizer, "normalizeMinDiff")(to_normalize)

            cnt = 0;
            for el in self.data:
                el[column_index] = normalized_column[cnt]
                cnt+=1

            self.normalizations[column_index] = normalization_info

        return self

    def denormalizeColumns(self):


        for column_index in xrange(0, len(self.data[0])):
            normalization_info = self.normalizations[column_index]
            to_denormalize  = [ numpy.float64(row[column_index]) for row in self.data ]

            # da migliorare
            if(normalization_info['type'] == Standardizer.NORMALIZATION_TYPE_MEAN_ABSMAX):
                denormalized_column = Standardizer.denormalize(to_denormalize, normalization_info['mean'], normalization_info['absmax'])
            else:
                denormalized_column = Standardizer.denormalizeMinDiff(to_denormalize, normalization_info['min'], normalization_info['diff'])


            cnt = 0;
            for el in self.data:
                el[column_index] = denormalized_column[cnt]
                cnt+=1

        return self






