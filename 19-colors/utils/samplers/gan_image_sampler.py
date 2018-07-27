#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from utils.misc import generate_noise
from utils.img_utils import rgb2gray as iu_rgb2gray
from utils.img_utils import split_and_downscale_lab
from utils.img_utils import LAB_RESCALE_FACTOR, AB_RESCALE_FACTOR

from skimage.color import rgb2lab, lab2rgb

from models.gan_lab import DISCRIMINATOR_OUTPUT_SHAPE

def gan_lab_image_sampler(datagen, batch_size, subset="training"):

    #  Model input: ([noise_input_a, noise_input_b, l_input, color_image_input], gan_prediction)

    for imgs in datagen.flow_from_directory('data/r_cropped', batch_size=batch_size, class_mode=None, subset=subset):

        g_y = []

        noises_a = []
        noises_b = []
        l_inputs = []
        original_colored_images = []

        for i in imgs:

            lab_image = rgb2lab(i / 255.0)

            l, ab = split_and_downscale_lab(lab_image)

            l_inputs.append(l)
            original_colored_images.append(ab)

            # the expected output is the original image
            g_y.append(np.ones( DISCRIMINATOR_OUTPUT_SHAPE ))


            noises_a.append(generate_noise(1, 256, 1)[0])
            noises_b.append(generate_noise(1, 256, 1)[0])



        yield([np.array(noises_a), np.array(noises_b), np.array(l_inputs), np.array(original_colored_images)], np.array(g_y))

RETURN_REAL_IMAGES = True
RETURN_GENERATED_IMAGES = False


def discriminator_lab_image_sampler_with_color_image(graph, datagen, generator, batch_size, subset="training", return_real_images=RETURN_REAL_IMAGES):

    for imgs in datagen.flow_from_directory('data/r_cropped', batch_size=batch_size, class_mode=None,  subset=subset, seed =1):
        g_y = []

        bw_image_inputs = []
        colored_image_input = []

        # RGB -> RGB1 -> LAB -> LAB1
        for i in imgs:
            bw_image = iu_rgb2gray(i)



            lab_image = rgb2lab(i / 255.0)


            # rescaled_lab = np.concatenate((l, a, b), axis=3)

            # range is now (0, 1), (-1, 1), (-1, 1)
            # rescaled_lab = lab_image / LAB_RESCALE_FACTOR


            # print("a: ", np.amin(rescaled_lab[:, :, 1])," - ",np.amax(rescaled_lab[:, :, 1]))
            # print("b: ", np.amin(rescaled_lab[:, :, 2])," - ",np.amax(rescaled_lab[:, :, 2]))

            # #AB range is now (0, 2)
            # rescaled_lab[:, :, 1] += 1.0
            # #AB range is now (0, 1)
            # rescaled_lab[:, :, 1] /= 2.0

            # print("a: ", np.amin(rescaled_lab[:, :, 1])," - ",np.amax(rescaled_lab[:, :, 1]))
            # print("b: ", np.amin(rescaled_lab[:, :, 2])," - ",np.amax(rescaled_lab[:, :, 2]))

            # exit()


            # l = rescaled_lab[:, :, 0]
            # l = l.reshape(256, 256, 1)

            l, ab = split_and_downscale_lab(lab_image)

            bw_image_inputs.append(l)

            # a = lab_image[:, :, 1]
            # b = lab_image[:, :, 2]

            #ab = lab_image[:, :, 1:]


            #original_colored_image_input.append(rescaled_lab)


            # generate data for REAL images
            #inputs
            if return_real_images == RETURN_REAL_IMAGES:
                # as a colored image, return the real one
                colored_image_input.append( ab )

                #output
                g_y.append(np.ones( DISCRIMINATOR_OUTPUT_SHAPE ))
            else:
                # generate data for FAKE images
                noise_a = generate_noise(1, 256, 1)
                noise_b = generate_noise(1, 256, 1)


                with graph.as_default():
                    prediction = generator.predict([noise_a, noise_b, l.reshape(1, 256, 256, 1)])[0]

                # prediction = prediction * AB_RESCALE_FACTOR
                # prediction = (lab2rgb(prediction) * 255.0).astype(np.uint8)
                # colored_image_input.append(prediction / 255.0 )
                colored_image_input.append(prediction)

                g_y.append(np.zeros( DISCRIMINATOR_OUTPUT_SHAPE ))

        yield([ np.array(bw_image_inputs), np.array(colored_image_input)], np.array(g_y))

