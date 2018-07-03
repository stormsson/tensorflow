""" GAN

Usage:
    gan.py  [-g MODEL] [-d MODEL] [-c MODEL] [--epochs=<e>] [--save-interval=<s>] [--train-generator|--train-discriminator] [--run-name=<run>]

Options:
    -g MODEL --generator=MODEL      specify generator model to restore
    -d MODEL --discriminator=MODEL  specify discriminator model to restore
    -c MODEL --combined-model-weights=MODEL specify combined model weights to restore
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

# KERAS stuff
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# TF stuff
import tensorflow as tf


from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb



import numpy as np


from bcolors import bcolors

# PROJECT STUFF
from gan_model import build_GAN_generator, build_GAN_discriminator, get_gan_optimizer, get_custom_objects_for_restoring

from utils.misc import generate_noise
from utils.saveload import restore_model, save_model
from utils.image_sampler import generator_image_sampler, discriminator_image_sampler, gan_image_sampler

RETURN_RGB_AS_Y = True
RETURN_LABELS_AS_Y = False

TOTAL_SAMPLES = 7060
TOTAL_SAMPLES = 200
TRAIN_SAMPLES = int(math.floor(TOTAL_SAMPLES * 0.8))
VALIDATION_SAMPLES = int(math.floor(TOTAL_SAMPLES * 0.2))



HALF_TRAIN_SAMPLES = int(math.floor(TRAIN_SAMPLES / 2 ))
HALF_VALIDATION_SAMPLES = int(math.floor(VALIDATION_SAMPLES / 2))



"""
support function to create y_labels  (expected classifier output) for the discriminator.
the discriminator should produce 1 if the provided image is real, 0 otherwise
the usage will be something like:

y_real_10, y_fake_10 = make_labels(10).
this will create an array of 10  labels with value "1" and 10 labels with value "0"
"""
def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])

def trainDiscriminator(generator, discriminator, EPOCHS, save_interval=5, run_name=""):
    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        validation_split=0.2)


    global HALF_TRAIN_SAMPLES
    global HALF_VALIDATION_SAMPLES


    callbacks = get_common_model_callbacks(run_name) + get_discriminator_model_callbacks(save_interval, "gan-discriminator-checkpoint")

    for x in xrange(0, EPOCHS):
        print("EPOCH %d of %d" % (x+1, EPOCHS))
        # train on REAL images
        print("Training on real samples")

        real_samples_loss = discriminator.fit_generator(
            discriminator_image_sampler(x_data_gen, generator, 1, "training"),
            validation_data=discriminator_image_sampler(x_data_gen, generator, 1, "validation"),
            epochs=1,
            steps_per_epoch= HALF_TRAIN_SAMPLES,
            validation_steps=HALF_VALIDATION_SAMPLES,
            callbacks=callbacks)

        # train on GENERATED images



        # for x, y in discriminator_image_sampler(x_data_gen, generator, HALF_TRAIN_SAMPLES, "training"):
        #     real_samples_loss = discriminator.train_on_batch(x, y)
        #     print("Discriminator loss on real images: ", real_samples_loss)
        #     break

        print("\nTraining on generated samples (progress will not be tracked :( )")
        # i can't use the generator model tf.graph object inside the python generator to create the images, so
        # first we create the batch of images, and then manually train on them

        for x, y in discriminator_image_sampler(x_data_gen, generator, HALF_TRAIN_SAMPLES, "training", False):
            generated_samples_loss = discriminator.train_on_batch(x, y)
            print("Discriminator loss on generated images: ", generated_samples_loss)
            break


        discriminator_loss = 0.5 * np.add(real_samples_loss.history["loss"][-1], generated_samples_loss[0])
        print("Discriminator Loss:", discriminator_loss)



    save_model(discriminator, "gan-discriminator-pretrained.h5","models/gan/")


def trainGenerator(generator, EPOCHS, save_interval=5, run_name=""):
    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        validation_split=0.2)

    global TRAIN_SAMPLES
    global VALIDATION_SAMPLES




    # TODO!!!! TESTARE QUESTO
    # callbacks = get_common_model_callbacks(run_name ) + get_generator_model_callbacks(save_interval,"gan-generator-checkpoint")

    callbacks = getModelCallbacks(run_name, save_interval,"gan-generator-checkpoint")

    generator.fit_generator(
        generator_image_sampler(x_data_gen, 1),
        validation_data=generator_image_sampler(x_data_gen, 1, "validation"),
        epochs=EPOCHS,
        steps_per_epoch= TRAIN_SAMPLES,
        validation_steps=VALIDATION_SAMPLES,
        callbacks=callbacks )

    save_model(generator, "gan-generator.h5","models/gan/")


def get_discriminator_model_callbacks(save_interval, checkpoint_prefix="gan-model-checkpoint"):
    global MODEL_SAVING_DIR

    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience= 7)
    reduceLROnPlateau_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr= 1e-9)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(MODEL_SAVING_DIR+checkpoint_prefix+"-{epoch:02d}-{val_loss:02f}.h5", period=save_interval, verbose=1)

    model_callbacks = [  reduceLROnPlateau_callback, model_checkpoint_callback ]
    return model_callbacks


# TODO !!!! TESTARE QUESTO

# def get_generator_model_callbacks(save_interval, checkpoint_prefix="gan-model-checkpoint"):
#     earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_l1_loss", patience= 7)
#     reduceLROnPlateau_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_l1_loss", factor=0.1, patience=5, min_lr= 1e-9)
#     model_checkpoint_callback = keras.callbacks.ModelCheckpoint("models/gan/"+checkpoint_prefix+"-{epoch:02d}-{val_l1_loss:02f}.h5", period=save_interval, verbose=1)

#     model_callbacks = [  reduceLROnPlateau_callback, model_checkpoint_callback]
#     return model_callbacks


"""
returns all the callbacks common for all models
"""
def get_common_model_callbacks(run_name):
    tb_log_dir = "./tensorboard"
    if(run_name):
        tb_log_dir+="/"+run_name

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)

    model_callbacks = [ tensorboard_callback ]
    return model_callbacks

def getModelCallbacks(run_name, save_interval,checkpoint_prefix="gan-model-checkpoint"):

    tb_log_dir = "./tensorboard"
    if(run_name):
        tb_log_dir+="/"+run_name

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)
    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_l1_loss", patience= 7)
    reduceLROnPlateau_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_l1_loss", factor=0.1, patience=5, min_lr= 1e-9)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint("models/gan/"+checkpoint_prefix+"-{epoch:02d}-{val_l1_loss:02f}.h5", period=save_interval, verbose=1)

    model_callbacks = [ tensorboard_callback, reduceLROnPlateau_callback, model_checkpoint_callback]
    return model_callbacks



from gan_test import compose_image
def train_gan(generator, discriminator, gan, EPOCHS, save_interval, run_name=""):

    global TRAIN_SAMPLES
    global VALIDATION_SAMPLES
    global HALF_TRAIN_SAMPLES
    global HALF_VALIDATION_SAMPLES
    global MODEL_SAVING_DIR



    graph = tf.get_default_graph()

    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        zoom_range=[1, 1.1],
        shear_range = 10,
        rotation_range=10,
        validation_split=0.2 )

    gan_callbacks =  get_discriminator_model_callbacks(save_interval, "gan-combined-model-checkpoint")

    for epoch in range(EPOCHS):
        fw = tf.summary.FileWriter(logdir="./tensorboard/"+run_name)

        print("EPOCH %d of %d" % (epoch, EPOCHS))
        print("Training Discriminator - REAL samples")

        real_samples_loss = discriminator.fit_generator(
            discriminator_image_sampler(graph, x_data_gen, generator, 1),
            validation_data=discriminator_image_sampler(graph, x_data_gen, generator, 1, subset="validation"),
            epochs=1,
            steps_per_epoch= HALF_TRAIN_SAMPLES,
            validation_steps=HALF_VALIDATION_SAMPLES,
            )

        print("Training Discriminator - GENERATED samples")

        generated_samples_loss = discriminator.fit_generator(
            discriminator_image_sampler(graph, x_data_gen, generator, 1, subset="training", return_real_images=False),
            validation_data=discriminator_image_sampler(graph, x_data_gen, generator, 1, subset="validation", return_real_images=False),
            epochs=1,
            steps_per_epoch= HALF_TRAIN_SAMPLES,
            validation_steps=HALF_VALIDATION_SAMPLES,
            )



        # for x, y in discriminator_image_sampler(x_data_gen, generator, HALF_TRAIN_SAMPLES, "training"):
        #     real_samples_loss = discriminator.train_on_batch(x, y)
        #     print("Discriminator loss on real images: ", real_samples_loss)

        #     summary = tf.Summary(value=[
        #         tf.Summary.Value(tag="gan_disc_real_loss", simple_value = real_samples_loss[0])
        #     ])
        #     fw.add_summary(summary, global_step=epoch)
        #     break

        # for x, y in discriminator_image_sampler(x_data_gen, generator, HALF_TRAIN_SAMPLES, "training", False):
        #     generated_samples_loss = discriminator.train_on_batch(x, y)
        #     print("Discriminator loss on generated images: ", generated_samples_loss)
        #     summary = tf.Summary(value=[
        #         tf.Summary.Value(tag="gan_disc_gen_loss", simple_value = generated_samples_loss[0])
        #     ])
        #     fw.add_summary(summary, global_step=epoch)
        #     break


        print("Training Generator (via cGAN)")


        gan_loss = gan.fit_generator(
            gan_image_sampler(x_data_gen, 1),
            validation_data=gan_image_sampler(x_data_gen, 1, "validation"),
            epochs=1,
            steps_per_epoch= TRAIN_SAMPLES,
            validation_steps=VALIDATION_SAMPLES,
            )


        #image_array = compose_image(generator, "data/r_cropped/real/part-05.png", MODEL_SAMPLES_DIR + "part-05-test-%d.jpg" % epoch, True)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="gan_disc_real_loss", simple_value = real_samples_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_disc_gen_loss", simple_value = generated_samples_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_loss", simple_value = gan_loss.history["loss"][-1]),
            tf.Summary.Value(tag="gan_val_loss", simple_value = gan_loss.history["val_loss"][-1]),
        ])

        fw.add_summary(summary, global_step=epoch)
        fw.flush()
        fw.close()


        compose_image(generator, "a2.jpg", MODEL_SAMPLES_DIR + "a2-test-%d.jpg" % epoch)
        compose_image(generator, "data/r_cropped/real/part-6350.jpg", MODEL_SAMPLES_DIR + "part-6350-test-%d.jpg" % epoch)
        compose_image(generator, "data/r_cropped/real/part-6248.jpg", MODEL_SAMPLES_DIR + "part-6248-test-%d.jpg" % epoch)


        if epoch > 0 and not (epoch % save_interval):
            save_model(generator, "gen-%d.h5" % epoch, MODEL_SAVING_DIR)
            save_model(discriminator, "disc-%d.h5" % epoch, MODEL_SAVING_DIR)
            gan.save_weights(MODEL_SAVING_DIR+"combined-%d-WEIGHTS.h5" % epoch)
            save_model(gan, "gan-full-gan-%d.h5" % epoch, MODEL_SAVING_DIR)



    save_model(generator, "gen-final.h5",MODEL_SAVING_DIR)
    save_model(discriminator, "disc-final.h5", MODEL_SAVING_DIR)
    gan.save_weights(MODEL_SAVING_DIR+"combined-%d-WEIGHTS-final.h5" % epoch)





    return


if __name__ == '__main__':


    arguments = docopt(__doc__, version="GAN 1.0")
    # print(arguments)
    # exit()

    EPOCHS = int(arguments["--epochs"])

    MODEL_SAVING_DIR = "models/gan/"+arguments["--run-name"]
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
        generator = build_GAN_generator()


    if(arguments["--train-generator"]):
        trainGenerator(generator, EPOCHS, int(arguments["--save-interval"]), arguments["--run-name"])
        exit()

    if arguments["--discriminator"]:
        discriminator = restore_model(arguments["--discriminator"], custom_objects)
    else:
        discriminator = build_GAN_discriminator()

    if(arguments["--train-discriminator"]):
        trainDiscriminator(generator, discriminator, EPOCHS, int(arguments["--save-interval"]), arguments["--run-name"])
        exit()


    # if arguments["--combined-model"]:
    #     combined_model = restore_model(arguments["--generator"], custom_objects)
    #     combined_model.summary()
    #     exit()
    # else:

    #     noise_input = Input(shape=(256, 256, 3,), name="combined_noise_input")
    #     bw_image_input = Input(shape=(256, 256, 3,), name="combined_bw_input")

    #     gen_img = generator([noise_input, bw_image_input])

    #     discriminator.trainable = False

    #     gan_prediction = discriminator([bw_image_input, gen_img])

    #     combined_model =  Model([noise_input, bw_image_input], gan_prediction)

    #     combined_model.compile(loss=["binary_crossentropy"], optimizer = get_gan_optimizer())



    noise_input = Input(shape=(256, 256, 3,), name="combined_noise_input")
    bw_image_input = Input(shape=(256, 256, 3,), name="combined_bw_input")

    gen_img = generator([noise_input, bw_image_input])

    discriminator.trainable = False

    gan_prediction = discriminator([bw_image_input, gen_img])

    combined_model =  Model([noise_input, bw_image_input], gan_prediction)

    if arguments["--combined-model-weights"]:
     combined_model.load_weights(arguments["--combined-model-weights"])

    combined_model.compile(loss=["binary_crossentropy"], optimizer = get_gan_optimizer())


    train_gan(generator, discriminator, combined_model, EPOCHS, int(arguments["--save-interval"]), arguments["--run-name"])