""" GAN

Usage:
    gan.py [-s] [-g MODEL] [-d MODEL] [--epochs=<e>]

Options:
    -g MODEL --generator=MODEL      specify generator model to restore
    -d MODEL --discriminator=MODEL  specify discriminator model to restore
    -s --skip-pretraining           skip discriminator pretraining
    --epochs=<e>                    training epochs [default: 100]
"""

from docopt import docopt

import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import sys
import random
import tensorflow as tf

from bcolors import bcolors

from model import restoreModel, build_GAN_generator, build_GAN_discriminator, l1_loss
from utils.misc import generate_noise



TRAIN_DIR = "data/r_cropped"


def getImageGenerator():
    # image generator
    datagen = ImageDataGenerator(
            rescale=1./255.0,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=True,
            validation_split = 0.2 )


    return datagen

   # test
    # color_me = []
    # for filename in os.listdir(TEST_DIR):
    #     color_me.append(img_to_array(load_img(TEST_DIR+filename)))
    # color_me = np.array(color_me, dtype=float)
    # color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    # color_me = color_me.reshape(color_me.shape+(1,))



     # lab_batch = rgb2lab(batch)
     #    X_batch = lab_batch[:,:,:,0]
     #    X_batch = X_batch.reshape(X_batch.shape+(1,))
     #    Y_batch = lab_batch[:,:,:,1:] / 128

       # Output colorizations
    # for i in range(len(output)):
    #     cur = np.zeros((256, 256, 3))
    #     cur[:,:,0] = color_me[i][:,:,0]
    #     cur[:,:,1:] = output[i]
    #     imsave("result/img_"+str(i)+".png", lab2rgb(cur))



"""
support function to create y_labels  (expected classifier output) for the discriminator.
the discriminator should produce 1 if the provided image is real, 0 otherwise
the usage will be something like:

y_real_10, y_fake_10 = make_labels(10).
this will create an array of 10  labels with value "1" and 10 labels with value "0"
"""
def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])



if __name__ == '__main__':
    arguments = docopt(__doc__, version="Gan 1.0")
    print(arguments)
    exit()


    dg = getImageGenerator()
    for x in xrange(1,10):
        cnt =0
        for img in  dg.flow_from_directory(TRAIN_DIR, batch_size=1, class_mode=None, subset="training", save_to_dir="tmp/train"):
            cnt = cnt+1
            asd = np.array(img, dtype=float) # shape: (1, 256, 256, 3)
            # print(asd.shape)
            # exit()
            grayscaled_rgb = gray2rgb(rgb2gray(asd))



            lab_img = rgb2lab(asd)


            imsave("cristo.png", lab2rgb(lab_img[0]))
            exit()

            if cnt > 0:
                break

            #cur = img_to_array(img)
            #cur = np.array(cur)
            #imsave("tmp/train/img_t-"+str(x)+".png", cur)


        cnt = 0
        for img in dg.flow_from_directory(TRAIN_DIR, batch_size=1, class_mode=None, subset="validation", save_to_dir="tmp/val"):
            cnt = cnt+1
            if cnt > 0:
                break


            #cur = img_to_array(img)
            #cur = np.array(cur)
            #imsave("tmp/val/img_v"+str(x)+".png", cur)


    exit()

    generator = build_GAN_generator()
    discriminator = build_GAN_discriminator()

    GAN = Sequential()
    GAN.add(generator)
    GAN.add(discriminator)

    LR = 0.0002
    B1 = 0.5
    DECAY = 0.0

    optim = keras.optimizers.Adam(lr=LR, beta_1=B1, beta_2=0.999, epsilon=1e-08, decay=DECAY)

    GAN.compile(optimizer=optim, loss=['mse', l1_loss])






 # with tf.device('/gpu:0'):



