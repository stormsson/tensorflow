""" GAN

Usage:
    gan.py [-p] [-g MODEL] [-d MODEL] [--epochs=<e>]

Options:
    -g MODEL --generator=MODEL      specify generator model to restore
    -d MODEL --discriminator=MODEL  specify discriminator model to restore
    -p --pretraining                only run discriminator pretraining
    --epochs=<e>                    training epochs [default: 100]
"""

from docopt import docopt

import random
import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import numpy as np
import tensorflow as tf

from bcolors import bcolors


from gan_model import build_GAN_generator, build_GAN_discriminator, get_gan_optimizer
from utils.misc import generate_noise, l1_loss
from utils.saveload import restore_model, save_model



TRAIN_DIR = "data/r_cropped"
DATAGEN_SEED = 1

def getImageGenerator():
    # image generator
    datagen = ImageDataGenerator(
            #rescale=1./255.0,
            horizontal_flip=True)
            #validation_split = 0.2 )


    return datagen
dg = getImageGenerator()

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


"""
from an rgb image array to a single L channel of LAB format
"""
def extract_L_channel_from_RGB(img):
    img = img / 255.0
    grayscaled_rgb = gray2rgb(rgb2gray(img))
    # Lab has range -127-128, so we divide by 128 to scale it down in the range -1/1
    lab = rgb2lab(grayscaled_rgb) / 128
    return lab[:,:,0]


def fetchDataForPretraining():
    SAMPLES = 400 # 540


    # prendo SAMPLES immagini dal subset di training
    # assegno loro la label 1

    # per ogni sample
    # lo converto in rgb-grey
    # lo converto in lab
    # estraggo il canale L
    # assegno la label 0
    # genero SAMPLES immagini col generator non allenato
    # assegno loro la label 0
    # passo tutto al discriminator

    real_images = []
    fake_images = []

    # generate real images from directory
    for imgs in  dg.flow_from_directory(TRAIN_DIR, batch_size=SAMPLES, class_mode=None, seed=DATAGEN_SEED):#, save_to_dir="tmp"):
        real_images = np.array(imgs)
        break


    # extract L channel and generate fake images with generator
    for img in real_images:

        # returns a (256, 256) shape
        lab_channel = np.array(extract_L_channel_from_RGB(img))

        # convert to (256, 256, 1) shape
        lab_channel = lab_channel.reshape(1, 256, 256, 1)

        fake_image = generator.predict(lab_channel)[0]
        fake_images.append(fake_image*255.0)
    fake_images = np.array(fake_images)


    #generate labels for both real and fake samples
    real_labels, fake_labels = make_labels(len(real_images))

    # data will be shuffled in the .fit method
    X = np.concatenate((real_images, fake_images))
    Y = np.concatenate((real_labels, fake_labels))

    return X, Y


def pretrainDiscriminator(generator, discriminator, EPOCHS):

    X, Y = fetchDataForPretraining()

    discriminator.fit(X, Y, validation_split= 0.2, verbose=1, epochs=EPOCHS, batch_size=1)

    save_model(discriminator, "gan-discriminator-pretrained","models/gan/")





if __name__ == '__main__':
    arguments = docopt(__doc__, version="Gan 1.0")
    print(arguments)


    EPOCHS = int(arguments["--epochs"])

    if arguments["--generator"]:
        generator = restore_model(arguments["--generator"])
    else:
        generator = build_GAN_generator()

    if arguments["--discriminator"]:
        discriminator = restore_model(arguments["--discriminator"])
    else:
        discriminator = build_GAN_discriminator()


    if(arguments["--pretraining"]):
        pretrainDiscriminator(generator, discriminator, EPOCHS)
        exit()


    GAN = Sequential()
    GAN.add(generator)
    GAN.add(discriminator)

    gan_optim = get_gan_optimizer()
    GAN.compile(optimizer=gan_optim, loss='mse')










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







