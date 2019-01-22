 #!/usr/bin/env python
# -*- coding: UTF-8 -*-

import keras
from keras.models import load_model
import numpy as np

from PIL import Image
from os.path import isfile, join
import matplotlib.pyplot as plt
from bcolors import bcolors

import sys

params = sys.argv

if(len(params) < 2):
    print(bcolors.FAIL + "Missing model filename. Usage python test_model [filename]"+ bcolors.ENDC)
    exit()

model_filename = params[1]
image_path = "images/286/BioID_0004.jpg"
if(len(params)==3):
    image_path = params[2]


model = False
try:
    model = load_model(model_filename)
except Exception as e:
    print(bcolors.FAIL + "Model "+model_filename+" not found"+ bcolors.ENDC)
    exit()

#rows, colsd, channels
IMG_WIDTH = 384
IMG_HEIGHT = 286

IMG_SIZE = ( 60, 45, 1 )
IMG_SIZE = ( 143, 192, 1 )
IMG_SIZE = ( IMG_HEIGHT, IMG_WIDTH, 1 )
print(model.summary())
exit()

img = Image.open(image_path)
img.load()
data = np.asarray(img, dtype="float32")
data = data/255

reshaped_data = np.reshape(data, IMG_SIZE)
coords = model.predict(np.array([reshaped_data]))


print("pre-mult coords: ",coords)

coords[0][0] = coords[0][0] * IMG_WIDTH
coords[0][1] = coords[0][1] * IMG_HEIGHT
coords[0][2] = coords[0][2] * IMG_WIDTH
coords[0][3] = coords[0][3] * IMG_HEIGHT
print(coords)
plt.imshow(data, cmap="gray")
plt.plot([coords[0][0], coords[0][2]],[coords[0][1], coords[0][3]],'o', color="red")
plt.show()