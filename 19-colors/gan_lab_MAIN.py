""" GAN

Usage:
    gan.py  [-f CONFIG] [-g MODEL] [-d MODEL] [-c MODEL] [--epochs=<e>] [--save-interval=<s>] [--train-generator|--train-discriminator] [--run-name=<run>] [--starting-epoch=<STARTING-EPOCH>]

Options:
    -f CONFIG --configuration-file=CONFIG  specify run configuration path, to auto configure run parameters
    -g MODEL --generator=MODEL      specify generator model to restore
    -d MODEL --discriminator=MODEL  specify discriminator model to restore
    -c MODEL --combined-model-weights=MODEL specify combined model weights to restore
    --starting-epoch=<STARTING-EPOCH>          step to start the training from (to append tensorboard data) [default: 1]
    -e --epochs=<e>                 training epochs [default: 100]
    -s --save-interval=<s>          save models after # epochs [default: 10]
    --train-generator               train only generator
    --train-discriminator           train only discriminator
    -r --run-name=<run>             name of the run in tensorboard [default: run]
"""

from docopt import docopt

import random
import math
import itertools
import os
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import gc

# KERAS stuff
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# TF stuff
import tensorflow as tf

# PROJECT STUFF

## MODEL
from models.gan_lab import buid_LAB_generator, build_GAN_LAB_discriminator, get_gan_optimizer
from models.gan_lab import DISCRIMINATOR_INPUT_SHAPE
from models.common import get_custom_objects_for_restoring
from models.losses import l1_loss, weighted_loss, merged_lab_discriminator_loss

## MISC
from bcolors import bcolors
from utils.misc import generate_noise
from utils.config import load_configuration_file, save_configuration_file
from utils.saveload import restore_model, save_model

## SAMPLERS
from utils.samplers.gan_image_sampler import gan_lab_image_sampler
from utils.samplers.gan_image_sampler import discriminator_lab_image_sampler_with_color_image

## TESTING
from gan_test import compose_image


RETURN_RGB_AS_Y = True
RETURN_LABELS_AS_Y = False

TOTAL_SAMPLES = 7060
TOTAL_SAMPLES = 500
# TOTAL_SAMPLES = 79

TRAIN_SAMPLES = int(math.floor(TOTAL_SAMPLES * 0.8))
VALIDATION_SAMPLES = int(math.floor(TOTAL_SAMPLES * 0.2))


HALF_TRAIN_SAMPLES = int(math.floor(TRAIN_SAMPLES / 2 ))
HALF_VALIDATION_SAMPLES = int(math.floor(VALIDATION_SAMPLES / 2))


arguments = {}


def train_gan(generator, discriminator, gan):

    global TRAIN_SAMPLES
    global VALIDATION_SAMPLES
    global HALF_TRAIN_SAMPLES
    global HALF_VALIDATION_SAMPLES
    global MODEL_SAVING_DIR

    global arguments


    save_interval = int(arguments["--save-interval"])
    run_name = arguments["--run-name"]
    tb_log_dir = "./tensorboard"
    if(run_name):
        tb_log_dir+="/"+run_name

    starting_epoch = int(arguments["--starting-epoch"])
    epochs = int(arguments["--epochs"])



    graph = tf.get_default_graph()

    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        zoom_range=[1, 1.1],
        shear_range = 10,
        rotation_range=10,
        validation_split=0.2 )

    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_graph=True, write_images=True)
    from utils.custom_tensorboard_callback import CustomTensorBoard
    discriminator_tensorboard_callback = CustomTensorBoard(log_dir=tb_log_dir, histogram_freq=5, write_graph=True, write_images=True, starting_epoch=starting_epoch)
    gan_tensorboard_callback = CustomTensorBoard(log_dir=tb_log_dir, histogram_freq=5, write_graph=True, write_images=True, starting_epoch=starting_epoch)


    for epoch in xrange(starting_epoch, epochs):
        fw = tf.summary.FileWriter(logdir="./tensorboard/"+run_name)

        print("EPOCH %d of %d" % (epoch, epochs))


        print("Training Discriminator - REAL samples")
        for x,y in discriminator_lab_image_sampler_with_color_image(graph, x_data_gen, generator, HALF_TRAIN_SAMPLES ):

            real_samples_loss = discriminator.fit(
                x, y,
                batch_size=1,
                epochs=1,
                validation_split=0.2,
                callbacks=[   ]  # discriminator_tensorboard_callback
            )
            break



        print("Training Discriminator - GENERATED samples")
        for x,y in discriminator_lab_image_sampler_with_color_image(graph, x_data_gen, generator, HALF_TRAIN_SAMPLES, return_real_images=False):
            generated_samples_loss = discriminator.fit(
                x, y,
                batch_size=1,
                epochs=1,
                validation_split=0.2,
                callbacks=[   ]# discriminator_tensorboard_callback

            )
            break

        disc_avg_loss = 0.5 * np.add(real_samples_loss.history["loss"][-1], generated_samples_loss.history["loss"][-1])


        print("Training Generator (via cGAN)")
        for x, y in gan_lab_image_sampler(x_data_gen, TRAIN_SAMPLES):


            gan_loss = gan.fit(
                {"gen_noise_input_a": x[0], "gen_noise_input_b": x[1], "gen_l_input":x[2], "discr_color_input": x[3]},
                y,
                batch_size=1,
                epochs=1,
                validation_split=0.2,
                callbacks = [   ]
            )
            break

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="gan_disc_real_loss", simple_value = real_samples_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_disc_gen_loss", simple_value = generated_samples_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_disc_avg_loss", simple_value = disc_avg_loss),
            tf.Summary.Value(tag="gan_loss", simple_value = gan_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_val_loss", simple_value = gan_loss.history["val_loss"][-1]),
        ])




        fw.add_summary(summary, global_step=epoch)
        fw.flush()
        fw.close()



        compose_image(generator, "a2.jpg", MODEL_SAMPLES_DIR + "ep-%d-a2-test.jpg" % epoch, fromLAB=True)
        compose_image(generator, "data/r_cropped/real/part-6350.jpg", MODEL_SAMPLES_DIR + "ep-%d-part-6350-test.jpg" % epoch, fromLAB=True)
        compose_image(generator, "data/r_cropped/real/part-6248.jpg", MODEL_SAMPLES_DIR + "ep-%d-part-6248-test.jpg" % epoch, fromLAB=True)


        if epoch > 0 and not (epoch % save_interval):
            save_model(generator, "gen-%d.h5" % epoch, MODEL_SAVING_DIR)
            save_model(discriminator, "disc-%d.h5" % epoch, MODEL_SAVING_DIR)
            gan.save_weights(MODEL_SAVING_DIR+"combined-%d-WEIGHTS.h5" % epoch)
            save_model(gan, "combined-%d.h5" % epoch, MODEL_SAVING_DIR)


    save_model(generator, "gen-final.h5",MODEL_SAVING_DIR)
    save_model(discriminator, "disc-final.h5", MODEL_SAVING_DIR)
    gan.save_weights(MODEL_SAVING_DIR+"combined-%d-WEIGHTS-final.h5" % epoch)





    return


if __name__ == '__main__':

    arguments = docopt(__doc__, version="GAN 1.0")

    if(arguments["--configuration-file"]):
        config = load_configuration_file(arguments["--configuration-file"])
        tmp = arguments.copy()
        tmp.update(config)
        arguments = tmp


    MODEL_SAVING_DIR = "data/models/gan/"+arguments["--run-name"]

    if not os.path.exists(MODEL_SAVING_DIR):
        os.makedirs(MODEL_SAVING_DIR)

    if not os.path.exists(MODEL_SAVING_DIR+"/samples"):
        os.makedirs(MODEL_SAVING_DIR+"/samples")

    MODEL_SAVING_DIR+="/"
    MODEL_SAMPLES_DIR = MODEL_SAVING_DIR+"samples/"

    custom_objects = get_custom_objects_for_restoring()


    if arguments["--generator"]:
        generator = restore_model(arguments["--generator"], custom_objects)
    else:
        generator = buid_LAB_generator()


    if(arguments["--train-generator"]):
        trainGenerator(generator, arguments)
        exit()

    if arguments["--discriminator"]:
        discriminator = restore_model(arguments["--discriminator"], custom_objects)
    else:
        discriminator = build_GAN_LAB_discriminator(compiled=True)

    if(arguments["--train-discriminator"]):
        trainDiscriminator(generator, discriminator, arguments)
        exit()


    noise_input_a = Input(shape=(256, 256, 1,), name="gen_noise_input_a")
    noise_input_b = Input(shape=(256, 256, 1,), name="gen_noise_input_b")
    l_input = Input(shape=(256, 256, 1,), name="gen_l_input")

    color_image_input = Input(shape=DISCRIMINATOR_INPUT_SHAPE, name="discr_color_input")

    generated_img = generator([noise_input_a, noise_input_b, l_input])

    discriminator.trainable = False

    # gan_prediction = discriminator([bw_image_input, generated_img])
    # gan_prediction = discriminator([color_image_input, generated_img])
    gan_prediction = discriminator([l_input, generated_img])

    combined_model =  Model([noise_input_a, noise_input_b, l_input, color_image_input], gan_prediction)

    if arguments["--combined-model-weights"]:
     combined_model.load_weights(arguments["--combined-model-weights"])

    # combined_model.compile(loss=["binary_c rossentropy"], optimizer = get_gan_optimizer())
    combined_model.compile(loss=[merged_lab_discriminator_loss(generated_img, color_image_input)], optimizer = get_gan_optimizer())


    train_gan(generator, discriminator, combined_model)