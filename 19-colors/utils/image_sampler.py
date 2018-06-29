#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from utils.misc import generate_noise
from utils.img_utils import rgb2gray as iu_rgb2gray


def generator_image_sampler(datagen, batch_size, subset="training"):

    for imgs in datagen.flow_from_directory('data/r_cropped', batch_size=batch_size, class_mode=None, seed=1, subset=subset):

        g_y = []

        noises = []
        bw_images = []
        for i in imgs:
            # the expected output is the original image
            g_y.append(i/255.0)


            noises.append(generate_noise(1, 256, 3)[0])
            bw_images.append(iu_rgb2gray(i))


        yield([np.array(noises), np.array(bw_images)], np.array(g_y))

        #yield({'noise_input': np.array(noises), 'bw_input': np.array(bw_images)}, {"img_output": np.array(g_y)})



RETURN_REAL_IMAGES = True
RETURN_GENERATED_IMAGES = False
def discriminator_image_sampler(datagen, generator, batch_size, subset="training", returned_images=RETURN_REAL_IMAGES):

    for imgs in datagen.flow_from_directory('data/r_cropped', batch_size=batch_size, class_mode=None, seed=1, subset=subset):

        g_y = []

        colored_images = []
        bw_images = []
        for i in imgs:
            bw_image = iu_rgb2gray(i)

            # black and white image is input for both real images and generated
            bw_images.append(bw_image)
            # generate data for REAL images
            #inputs
            if returned_images == RETURN_REAL_IMAGES:
                # as a colored image, return the real one
                colored_images.append(i/255.0)
                #output
                g_y.append(np.ones(1))
            else:
            # generate data for FAKE images

            noise = generate_noise(1, 256, 3)[0]
            colored_images.append(generator.predict([noise, bw_image])[0])
            g_y.append(np.zeros(1))


        yield([ np.array(bw_images), np.array(colored_images)], np.array(g_y))