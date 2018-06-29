""" GAN

Usage:
    gan.py [-p] [-g MODEL] [-d MODEL] [--epochs=<e>] [--save-interval=<s>] [--train-generator] [--run-name=<run>]

Options:
    -g MODEL --generator=MODEL      specify generator model to restore
    -d MODEL --discriminator=MODEL  specify discriminator model to restore
    -p --pretraining                only run discriminator pretraining
    -e --epochs=<e>                 training epochs [default: 100]
    -s --save-interval=<s>          save models after # epochs [default: 10]
    --train-generator               train only generator
    -r --run-name=<run>             name of the run in tensorboard [default: run]
"""

from docopt import docopt

import random
import math
import itertools

import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
import tensorflow as tf

from bcolors import bcolors


from gan_model import build_GAN_generator, build_GAN_discriminator, get_gan_optimizer, build_complex_GAN_generator
from utils.misc import generate_noise, l1_loss, l2_loss, set_trainable
from utils.saveload import restore_model, save_model
from utils.image_sampler import generator_image_sampler

RETURN_RGB_AS_Y = True
RETURN_LABELS_AS_Y = False



def getImageGenerator():
    # image generator
    datagen = ImageDataGenerator(
            #rescale=1./255.0,
            horizontal_flip=True)
            #validation_split = 0.2 )


    return datagen


"""
support function to create y_labels  (expected classifier output) for the discriminator.
the discriminator should produce 1 if the provided image is real, 0 otherwise
the usage will be something like:

y_real_10, y_fake_10 = make_labels(10).
this will create an array of 10  labels with value "1" and 10 labels with value "0"
"""
def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


"""
generate a X,Y dataset composed by both real elements read from input folders
and generated fake pictures from the generator
"""
def generateSamplesForDiscriminator(samples_dir, generator, real_images):


    fake_images = []

    # generate fake images with generator
    for img in real_images:
        grayscaled_rgb = gray2rgb(rgb2gray(1.0/255*img))
        grayscaled_rgb = np.array(grayscaled_rgb)


        fake_image = generator.predict(grayscaled_rgb.reshape(1, IMG_SIZE, IMG_SIZE, 3))[0]
        fake_images.append(fake_image*255.0)
    fake_images = np.array(fake_images)


    #generate labels for both real and fake samples
    real_labels, fake_labels = make_labels(len(real_images))



    try:
        # data will be shuffled in the .fit method
        X = np.concatenate((real_images, fake_images))
        Y = np.concatenate((real_labels, fake_labels))
    except Exception as e:
        print("Len: real: %d, fake: %d" %(len(real_images),len(fake_images)))
        raise e

    return X, Y


# def pretrainDiscriminator(generator, discriminator, EPOCHS, real_images):

#     X, Y = generateSamplesForDiscriminator(TRAIN_DIR, generator, real_images)

#     discriminator.fit(X, Y, validation_split= 0.2, verbose=1, epochs=EPOCHS, batch_size=1)

#     save_model(discriminator, "gan-discriminator-pretrained.h5","models/gan/")

def trainDiscriminator(generator, discriminator, EPOCHS, save_interval=5, run_name=""):
    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        validation_split=0.2)

    TRAIN_SAMPLES = 540
    VALIDATION_SAMPLES = 135

    half_train_batch = math.floor(TRAIN_SAMPLES)
    half_validation_batch = math.floor(VALIDATION_SAMPLES)

    discriminator.fit_generator(
        discriminator_image_sampler(x_data_gen, generator, 1, "training"),
        validation_data=discriminator_image_sampler(x_data_gen, generator, 1, "validation"),
        epochs=EPOCHS,
        steps_per_epoch= TRAIN_SAMPLES,
        validation_steps=VALIDATION_SAMPLES,
        callbacks=getModelCallbacks(run_name, save_interval,"gan-generator-checkpoint") )

    save_model(discriminator, "gan-discriminator-pretrained.h5","models/gan/")



def trainGenerator(generator, EPOCHS, save_interval=5, run_name=""):
    x_data_gen = ImageDataGenerator(
        #rescale=1./255.0,
        horizontal_flip=True,
        validation_split=0.2)

    TRAIN_SAMPLES = 540
    VALIDATION_SAMPLES = 135

    generator.fit_generator(
        generator_image_sampler(x_data_gen, 1),
        validation_data=generator_image_sampler(x_data_gen, 1, "validation"),
        epochs=EPOCHS,
        steps_per_epoch= TRAIN_SAMPLES,
        validation_steps=VALIDATION_SAMPLES,
        callbacks=getModelCallbacks(run_name, save_interval,"gan-generator-checkpoint") )

    save_model(generator, "gan-generator.h5","models/gan/")



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


if __name__ == '__main__':


    TRAIN_DIR = "data/r_cropped"
    DATAGEN_SEED = 1
    VALIDATION_SPLIT = 0.2

    IMG_SIZE = 256
    SAMPLES = 670
    real_images = []
    dg = getImageGenerator()


    arguments = docopt(__doc__, version="GAN 1.0")
    # print(arguments)
    # exit()

    EPOCHS = int(arguments["--epochs"])

    custom_objects = {
        "l1_loss": l1_loss,
        "l2_loss": l2_loss
    }

    if arguments["--generator"]:
        generator = restore_model(arguments["--generator"], custom_objects)
    else:
        generator = build_GAN_generator()
        # generator = build_complex_GAN_generator()

    if(arguments["--train-generator"]):
        real_images=[]
        trainGenerator(generator, EPOCHS, int(arguments["--save-interval"]), arguments["--run-name"])
        exit()

    if arguments["--discriminator"]:
        discriminator = restore_model(arguments["--discriminator"], custom_objects)
    else:
        discriminator = build_GAN_discriminator()



    if(arguments["--pretraining"]):
        real_images = generateRealImages(TRAIN_DIR, SAMPLES , DATAGEN_SEED)
        trainDiscriminator(generator, discriminator, EPOCHS, int(arguments["--save-interval"]), arguments["--run-name"])
        exit()




    GAN = Sequential()
    GAN.add(generator)
    GAN.add(discriminator)

    gan_optim = get_gan_optimizer()
    GAN.compile(optimizer=gan_optim, loss='binary_crossentropy', metrics=["accuracy", "mae"])


    d_loss= []
    g_loss = []

    #get samples for generator (they do not change between epochs)
    g_X, _ = generateSamplesForGenerator(real_images)

    # generate the samples and give them the 1.0 label, because we expect them to be generated to be percevied as
    # TRUE paintings
    g_Y, _ = make_labels(len(g_X))


    for epochs_counter in xrange(1, EPOCHS+1):


        print("TRAINING EPOCH %d / %d " % (epochs_counter, EPOCHS))


        # train generator by training the GAN; we disable discriminator training
        set_trainable(discriminator, False)
        GAN.compile(optimizer=gan_optim, loss='binary_crossentropy', metrics=["accuracy", "mae"])

        # fit
        print("Training GAN")
        g_loss.append(GAN.fit(g_X, g_Y, epochs=1, validation_split=VALIDATION_SPLIT, batch_size=1))

        # train discriminator
        set_trainable(discriminator, True)
        # GAN.compile(optimizer=gan_optim, loss='binary_crossentropy', metrics=["accuracy", "mae"])
        # get samples

        # at each epoch we need to regenerate the images from the generator
        d_X, d_Y = generateSamplesForDiscriminator(TRAIN_DIR, generator, real_images)

        # fit
        print("Training discriminator")
        d_loss.append(discriminator.fit(d_X, d_Y, epochs=1, validation_split=VALIDATION_SPLIT, batch_size=1))

        print("Discriminator loss:", d_loss[-1])
        print("GAN loss:", g_loss[-1])


        if not (epochs_counter) % int(arguments["--save-interval"]):
            print("Epoch %d checkpoint. saving models" % epochs_counter)

            save_model(generator, "gan-generator.h5","models/gan/")
            save_model(discriminator, "gan-discriminator.h5","models/gan/")

            #save models
            pass

    exit()


#lab_img = rgb2lab(img_rgb_array)

#imsave("cristo.png", lab2rgb(lab_img[0]))
#exit()








