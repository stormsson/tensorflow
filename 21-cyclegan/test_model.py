#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from keras.models import load_model
from PIL import Image
import numpy as np


generator_AtoB = load_model("models/generator_AtoB.h5")
generator_BtoB = load_model("models/generator_BtoA.h5")

A = Image.open("A.jpg")
B = Image.open("B.jpg")

A =np.array(A)
B = np.array(B)

A = A/127.5 -1
B = B/127.5 -1


out_b = generator_AtoB.predict(B)
pippo = generator_AtoB.predict(out_b)
print(pippo.shape)
exit()
x = Image.fromarray(pippo)
