#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# https://hardikbansal.github.io/CycleGANBlog/

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, multiply, add as kadd
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from keras.layers import LeakyReLU, ReLU
from keras.layers import Activation

from keras.preprocessing.image import ImageDataGenerator

import keras
import numpy as np

ngf = 32 # Number of filters in first layer of generator
ndf = 64 # Number of filters in first layer of discriminator
BATCH_SIZE = 1 # batch_size
pool_size = 50 # pool_size
IMG_WIDTH = 256 # Imput image will of width 256
IMG_HEIGHT = 256 # Input image will be of height 256
IMG_DEPTH = 3 # RGB format

DISCRIMINATOR_ITERATIONS = 1
SAVE_IMAGES_INTERVAL = 5

ITERATIONS = 10
FAKE_POOL_SIZE=25
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)


def resnet_block(num_features):

    block = Sequential()
    block.add(Conv2D(num_features, kernel_size=3, strides=1, padding="SAME"))
    block.add(BatchNormalization())
    block.add(ReLU())
    block.add(Conv2D(num_features, kernel_size=3, strides=1, padding="SAME"))
    block.add(BatchNormalization())
    block.add(ReLU())


    resblock_input = Input(shape=(64, 64, 256))
    conv_model = block(resblock_input)

    # _sum = Sequential()
    # _sum.add(Add([resblock_input, conv_model]))
      #      model_input = multiply([noise_input, bw_image_input])
    _sum = kadd([resblock_input, conv_model])

    composed =  Model(inputs=[resblock_input], outputs=_sum)
    return composed


def discriminator( f=4, name=None):
    d = Sequential()
    d.add(Conv2D(ndf, kernel_size=f, strides=2, padding="SAME", name="discr_conv2d_1"))
    d.add(BatchNormalization())
    d.add(LeakyReLU(0.2))
    d.add(Conv2D(ndf * 2, kernel_size=f, strides=2, padding="SAME", name="discr_conv2d_2"))
    d.add(BatchNormalization())
    d.add(LeakyReLU(0.2))
    d.add(Conv2D(ndf * 4, kernel_size=f, strides=2, padding="SAME", name="discr_conv2d_3"))
    d.add(BatchNormalization())
    d.add(LeakyReLU(0.2))
    d.add(Conv2D(ndf * 8, kernel_size=f, strides=2, padding="SAME", name="discr_conv2d_4"))
    d.add(BatchNormalization())
    d.add(LeakyReLU(0.2))
    d.add(Conv2D(1, kernel_size=f, strides=1, padding="SAME", name="discr_conv2d_out"))

    # d.add(Activation("sigmoid"))


    model_input = Input(shape=INPUT_SHAPE)

    decision  = d(model_input)

    composed = Model(model_input, decision)
    # print(d.output_shape)
    # d.summary()

    return composed

def generator(name=None):

    g = Sequential()
    # ENCODER
    g.add(Conv2D(ngf, kernel_size=7,
            strides=1,
            # activation='relu',
            padding='SAME',
            input_shape=INPUT_SHAPE,
            name="encoder_0" ))


    g.add(Conv2D(64*2, kernel_size=3,
            strides=2,
            padding='SAME',
            name="encoder_1" ))
    # output shape = (128, 128, 128)


    g.add(Conv2D(64*4, kernel_size=3,
            padding="SAME",
            strides=2,))
    # output shape = (64, 64, 256)

    # END ENCODER


    # TRANSFORM

    g.add(resnet_block(64*4))
    g.add(resnet_block(64*4))
    g.add(resnet_block(64*4))
    g.add(resnet_block(64*4))
    g.add(resnet_block(64*4))

    # o_r1 = resnet_block(g, num_features=64*4)
    # o_r2 = build_resnet_block(o_r1, num_features=64*4)
    # o_r3 = build_resnet_block(o_r2, num_features=64*4)
    # o_r4 = build_resnet_block(o_r3, num_features=64*4)
    # o_r5 = build_resnet_block(o_r4, num_features=64*4)

    # generator = build_resnet_block(o_r5, num_features=64*4)

    # END TRANSFORM
    # generator.shape = (64, 64, 256)

    # DECODER

    g.add(Conv2DTranspose(ngf*2,kernel_size=3, strides=2, padding="SAME"))
    g.add(Conv2DTranspose(ngf*2,kernel_size=3, strides=2, padding="SAME"))

    g.add(Conv2D(3,kernel_size=7, strides=1, padding="SAME"))

    # END DECODER

    model_input = Input(shape=INPUT_SHAPE)
    generated_image = g(model_input)

    composed = Model(model_input, generated_image, name=name)
    return composed


def fromMinusOneToOne(x):
    return x/127.5 -1


def createImageGenerator( subset="train", data_type="A", batch_size=1, pp=None):

    # we create two instances with the same arguments
    data_gen_args = dict(
                         # rescale = 1./127.5,
                         rotation_range=5.,
                         preprocessing_function= pp,
                         # width_shift_range=0.1,
                         # height_shift_range=0.1,
                         zoom_range=0.1)

    image_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_directory=subset+data_type
    print('data/vangogh2photo/'+image_directory)
    image_generator = image_datagen.flow_from_directory(
        'data/vangogh2photo/'+image_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed)

    return image_generator


if __name__ == '__main__':

    model = Sequential()

    generator_AtoB = generator(name="gen_A")
    generator_BtoA = generator(name="gen_B")

    discriminator_A = discriminator(name="disc_A")
    discriminator_B = discriminator(name="disc_B")


    # input_A = Input(batch_shape=(batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="input_A")
    input_A = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="input_A")
    generated_B = generator_AtoB(input_A)
    discriminator_generated_B = discriminator_B(generated_B)
    cyc_A = generator_BtoA(generated_B)


    input_B = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="input_B")
    generated_A = generator_BtoA(input_B)
    discriminator_generated_A = discriminator_A(generated_A )
    cyc_B = generator_AtoB(generated_A)


    # generated_A = generator(input_B, name="generated_A")

    # discriminator_original_A = discriminator(input_A, name="discriminator_A_given_A")
    # discriminator_original_B = discriminator(input_B, name="discriminator_B_given_B")


    # discriminator_generated_A = discriminator(generated_A, name="discriminator_generated_A")

    # il generator "B to A"
    # cyc_B = generator(generated_A, "cyc_B")


    # Loss per Discriminator A (D_A) si occupa di
    # - identificare l'immagine "A" originale   => 1
    # - identificare l'immagine "A" generata    => 0

    # D_A_loss_1 = tf.reduce_mean(tf.squared_difference(discriminator_original_A,1))
    # D_A_loss_2 = tf.reduce_mean(tf.square(discriminator_generated_A))
    # D_A_loss = (D_A_loss_1 + D_A_loss_2)/2

    # Loss per Discriminator B  (D_B) si occupa di
    # - identificare l'immagine "B" originale   => 1
    # - identificare l'immagine "B" generata    => 0

    # D_B_loss_2 = tf.reduce_mean(tf.square(discriminator_generated_B))
    # D_B_loss_1 = tf.reduce_mean(tf.squared_difference(discriminator_original_B,1))
    # D_B_loss = (D_B_loss_1 + D_B_loss_2)/2


    # Loss per Generator A si occupa di:
    # - identificare l'immagine "A"

    # differenza tra "capacità di distinguere immagine A originale" e 1 (per immagine A originale)
    # g_loss_B_1 = tf.reduce_mean(tf.squared_difference(discriminator_generated_A,1))

    # g_loss_A_1 = tf.reduce_mean(tf.squared_difference(discriminator_generated_B,1))
    # differenza tra "capacità di distinguere immagine B originale" e 1 (per immagine B originale)

    # cyc_loss è una somma di errori, come tale l'obiettivo è minimizzare il valore
    # cyc_A
    # cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))

    #
    # g_loss_A = g_loss_A_1 + 10*cyc_loss
    # g_loss_B = g_loss_B_1 + 10*cyc_loss



    ### GENERATOR TRAINING
    optim = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)



    generator_trainer =  Model([input_A, input_B],
                     [discriminator_generated_B,   discriminator_generated_A,
                     cyc_A,      cyc_B,
                     generated_B,     generated_A ]
                     )

    # cyclic error is increased, because it's more important
    cyclic_weight_multipier = 10

    losses =         [ "MSE", "MSE", "MAE",                   "MAE",                    "MAE", "MAE"]
    losses_weights = [ 1,     1,     cyclic_weight_multipier, cyclic_weight_multipier,  1,     1    ]

    generator_trainer.compile(optimizer=optim, loss = losses, loss_weights=losses_weights)



    ### DISCRIMINATOR TRAINING

    disc_optim = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    real_A = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="in_real_A")
    real_B = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="in_real_B")

    generated_A = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="in_gen_A")
    generated_B = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="in_gen_B")

    discriminator_real_A = discriminator_A(real_A)
    discriminator_generated_A = discriminator_A(generated_A)
    discriminator_real_B =  discriminator_B(real_B)
    discriminator_generated_B = discriminator_B(generated_B)

    disc_trainer = Model([real_A, generated_A, real_B, generated_B],
                         [  discriminator_real_A,
                            discriminator_generated_A,
                            discriminator_real_B,
                            discriminator_generated_B] )


    disc_trainer.compile(optimizer=disc_optim, loss = 'MSE')


    #########
    ##
    ## TRAINING
    ##
    #########


    fake_A_pool = []
    fake_B_pool = []


    ones = np.ones((BATCH_SIZE,)+ generator_trainer.output_shape[0][1:])
    zeros = np.zeros((BATCH_SIZE,)+ generator_trainer.output_shape[0][1:])



    train_A_image_generator = createImageGenerator("train", "A")
    # print(train_A_image_generator.next())
    # for c in train_A_image_generator:
    #     print(c)
    #     exit()
    # exit()

    train_B_image_generator = createImageGenerator("train", "B")
    test_A_image_generator = createImageGenerator("test", "A")
    test_B_image_generator = createImageGenerator("test", "B")

    it = 1
    while it  < ITERATIONS:
        print("Iteration %d " % it)



        # THIS ONLY WORKS IF BATCH SIZE == 1
        real_A = train_A_image_generator.next()
        real_B = train_B_image_generator.next()

        fake_A_pool.extend(generator_BtoA.predict(real_B))
        fake_B_pool.extend(generator_AtoB.predict(real_A))

        #resize pool
        fake_A_pool = fake_A_pool[-FAKE_POOL_SIZE:]
        fake_B_pool = fake_B_pool[-FAKE_POOL_SIZE:]

        fake_A = [ fake_A_pool[ind] for ind in np.random.choice(len(fake_A_pool), size=(BATCH_SIZE,), replace=False) ]
        fake_B = [ fake_B_pool[ind] for ind in np.random.choice(len(fake_B_pool), size=(BATCH_SIZE,), replace=False) ]

        fake_A = np.array(fake_A)
        fake_B = np.array(fake_B)


        for x in range(0, DISCRIMINATOR_ITERATIONS):
            _, D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B = disc_trainer.train_on_batch(
                [real_A, fake_A, real_B, fake_B],
                [zeros, ones * 0.9, zeros, ones * 0.9] )


        print("Discriminator loss:")
        print("Real A: %s, Fake A: %s, Real B: %s, Fake B: %s " % ( D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B))
        exit()
        it+=1

     # lmbd = 10.0,
     #    idloss=1.0,

     #    # optimizers
     #    lr = 0.0002,            ## initial learning rate for adam
     #    beta1 = 0.5,            ## momentum term of
