#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

"""
generate a fake image
"""
def generate_noise(amount=1, size=256, channels=3 ):
    noise = np.random.uniform(0, 1, (amount, size, size, channels))
    return noise