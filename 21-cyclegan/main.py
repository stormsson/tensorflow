#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# https://hardikbansal.github.io/CycleGANBlog/

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, multiply, add as kadd
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose
from keras.layers import LeakyReLU, ReLU


import keras

ngf = 32 # Number of filters in first layer of generator
ndf = 64 # Number of filters in first layer of discriminator
batch_size = 1 # batch_size
pool_size = 50 # pool_size
img_width = 256 # Imput image will of width 256
img_height = 256 # Input image will be of height 256
img_depth = 3 # RGB format

INPUT_SHAPE = (img_width, img_height, img_depth)


def resnet_block(num_features):

    block = Sequential()
    block.add(Conv2D(num_features, kernel_size=3, strides=1, padding="SAME"))
    block.add(BatchNormalization())
    block.add(ReLU(0.2))
    block.add(Conv2D(num_features, kernel_size=3, strides=1, padding="SAME"))
    block.add(BatchNormalization())
    block.add(ReLU(0.2))


    resblock_input = Input(shape=(64, 64, 256))
    conv_model = block(resblock_input)

    # _sum = Sequential()
    # _sum.add(Add([resblock_input, conv_model]))
      #      model_input = multiply([noise_input, bw_image_input])
    _sum = kadd([resblock_input, conv_model])

    composed =  Model(inputs=[resblock_input], outputs=_sum)
    return composed


def discriminator(input_disc, f=4, name=None):
    d = Sequential()
    d.add(Conv2D(ndf, kernel_size=f, strides=2, padding="SAME"))
    d.add(Conv2D(ndf * 2, kernel_size=f, strides=2, padding="SAME"))
    d.add(Conv2D(ndf * 4, kernel_size=f, strides=2, padding="SAME"))
    d.add(Conv2D(ndf * 8, kernel_size=f, strides=2, padding="SAME"))
    d.add(Conv2D(1, kernel_size=f, strides=1, padding="SAME"))

    out = d(input_disc)
    return out

def generator(input_gen, name=None):

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



    out = g(input_gen)
    return out

if __name__ == '__main__':

    model = Sequential()

    input_A = Input(batch_shape=(None, img_width, img_height, img_depth), name="input_A")
    # input_A = Input(batch_shape=(batch_size, img_width, img_height, img_depth), name="input_A")
    input_B = Input(batch_shape=(None, img_width, img_height, img_depth), name="input_B")


    generator_AtoB = generator(input_A, name="generator_AtoB")
    generator_BtoA = generator(input_B, name="generator_BtoA")

    discriminator_original_A = discriminator(input_A, name="discriminator_A")
    discriminator_original_B = discriminator(input_B, name="discriminator_B")


    discriminator_generated_AtoB = discriminator(generator_AtoB, name="discriminator_generated_B")
    discriminator_generated_BtoA = discriminator(generator_BtoA, name="discriminator_generated_A")

    # il generator "B to A"
    cyc_A = generator(generator_AtoB, "generator_BtoA")
    cyc_B = generator(generator_BtoA, "generator_AtoB")




    # Loss per Discriminator A (D_A) si occupa di
    # - identificare l'immagine "A" originale   => 1
    # - identificare l'immagine "A" generata    => 0

    # D_A_loss_1 = tf.reduce_mean(tf.squared_difference(discriminator_original_A,1))
    # D_A_loss_2 = tf.reduce_mean(tf.square(discriminator_generated_BtoA))
    # D_A_loss = (D_A_loss_1 + D_A_loss_2)/2

    # Loss per Discriminator A  (D_B) si occupa di
    # - identificare l'immagine "B" originale   => 1
    # - identificare l'immagine "B" generata    => 0

    # D_B_loss_2 = tf.reduce_mean(tf.square(discriminator_generated_AtoB))
    # D_B_loss_1 = tf.reduce_mean(tf.squared_difference(discriminator_original_B,1))
    # D_B_loss = (D_B_loss_1 + D_B_loss_2)/2


    # Loss per Generator A si occupa di:
    # - identificare l'immagine "A"

    # differenza tra "capacità di distinguere immagine A originale" e 1 (per immagine A originale)
    # g_loss_B_1 = tf.reduce_mean(tf.squared_difference(discriminator_generated_BtoA,1))

    # g_loss_A_1 = tf.reduce_mean(tf.squared_difference(discriminator_generated_AtoB,1))
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
                     [discriminator_generated_AtoB,   discriminator_generated_BtoA,
                     cyc_A,      cyc_B,
                     generator_AtoB,     generator_BtoA ]
                     )

    # cyclic error is increased, because it's more important
    cyclic_weight_multipier = 10

    losses =         [ "MSE", "MSE", "MAE",                   "MAE",                    "MAE", "MAE"]
    losses_weights = [ 1,     1,     cyclic_weight_multipier, cyclic_weight_multipier,  1,     1    ]

    generator_trainer.compile(optimizer=optim, loss = losses, loss_weights=losses_weights)



    ### DISCRIMINATOR TRAINING

    disc_optim = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    real_A = Input(batch_shape=(None, img_width, img_height, img_depth), name="in_real_A")
    real_B = Input(batch_shape=(None, img_width, img_height, img_depth), name="in_real_B")

    generated_A = Input(batch_shape=(None, img_width, img_height, img_depth), name="in_gen_A")
    generated_B = Input(batch_shape=(None, img_width, img_height, img_depth), name="in_gen_B")

    disc_trainer = Model([real_A, generated_A, real_B, generated_B],
                         [  discriminator_original_A,
                            discriminator_generated_A,
                            discriminator_original_B,
                            discriminator_generated_B] )





     # lmbd = 10.0,
     #    idloss=1.0,

     #    # optimizers
     #    lr = 0.0002,            ## initial learning rate for adam
     #    beta1 = 0.5,            ## momentum term of