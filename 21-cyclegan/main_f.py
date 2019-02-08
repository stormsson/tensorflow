#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# https://hardikbansal.github.io/CycleGANBlog/
import sys
import time
import pickle
import tensorflow as tf
import numpy as np
import keras
import random

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout
from keras.layers import multiply
from keras.layers.merge import Add
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, ReLU
from keras.layers import Activation
from keras.initializers import RandomNormal

from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

from keras_contrib.layers.normalization import InstanceNormalization
from keras.utils import multi_gpu_model


from custom_layers import ReflectionPadding2D


# NET PARAMETERS
ngf = 64 # Number of filters in first layer of generator
ndf = 64 # Number of filters in first layer of discriminator
BATCH_SIZE = 1 # batch_size
pool_size = 50 # pool_size
IMG_WIDTH = 256 # Imput image will of width 256
IMG_HEIGHT = 256 # Input image will be of height 256
IMG_DEPTH = 3 # RGB format
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)

USE_IDENTITY_LOSS = False

IMAGE_RANGE_0_255 = False


# TRAINING PARAMETERS
ITERATIONS = 5500000
DISCRIMINATOR_ITERATIONS = 1
SAVE_IMAGES_INTERVAL = 100

SAVE_MODEL_INTERVAL = 1000

FAKE_POOL_SIZE=50

# DATASET="vangogh2photo"
DATASET="horse2zebra"


resnet_block_cnt = 1

def resnet_block(model_input, num_features):
    global resnet_block_cnt
    with K.name_scope('resnet_block_%s' % resnet_block_cnt):

        x = ReflectionPadding2D()(model_input)
        x = Conv2D(num_features, kernel_size=3, strides=1, padding="valid")(x)
        x = InstanceNormalization(axis=3,epsilon=1e-5, gamma_initializer=RandomNormal(1., 0.02))(x)
        x = Activation("relu")(x)


        x = ReflectionPadding2D( )(x)
        x = Conv2D(num_features, kernel_size=3, strides=1, padding="valid")(x)
        x = InstanceNormalization(axis=3,epsilon=1e-5, gamma_initializer=RandomNormal(1., 0.02))(x)

        _sum = Add()([model_input, x])


    resnet_block_cnt+=1
    return _sum

def scaleup(input_tensor, ngf):
    x = UpSampling2D(size=2)(input_tensor)
    x = Conv2D(ngf, kernel_size=3, strides=1, padding="same")(x)

    return x

def discriminator(input_shape, f=4, name=None):


    with K.name_scope(name):

        model_input = Input(shape=input_shape, name=name+"_input")
        x = Conv2D(ndf, kernel_size=f, strides=2, padding="same")(model_input)
        x = LeakyReLU(0.2)(x)

        # ===
        x = Conv2D(ndf * 2, kernel_size=f, strides=2, padding="same")(x)
        # x = BatchNormalization(axis=3)(x)
        x = InstanceNormalization(axis=3)(x)
        #x = Dropout(0.3)(x)
        x = LeakyReLU(0.2)(x)


        x = Conv2D(ndf * 4, kernel_size=f, strides=2, padding="same")(x)
        # x = BatchNormalization(axis=3)(x)
        x = InstanceNormalization(axis=3)(x)
        #x = Dropout(0.3)(x)
        x = LeakyReLU(0.2)(x)


        x = Conv2D(ndf * 8, kernel_size=f, strides=2, padding="same")(x)
        # x = BatchNormalization(axis=3)(x)
        x = InstanceNormalization(axis=3)(x)

        x = LeakyReLU(0.2)(x)

        # ===

        x = Conv2D(1, kernel_size=f, strides=1, padding="same", name=name+"_out_layer")(x)

        # x = Activation('sigmoid')(x)

        # print(d.output_shape)
        # d.summary()
    composed = Model(model_input, x, name=name)
    composed.summary()
    # exit()

    return composed

def generator(input_shape, name):

    with K.name_scope(name):

        # ENCODER
        model_input = Input(shape=input_shape, name=name+"_input")

        x = ReflectionPadding2D(padding=(3,3))(model_input)
        x = Conv2D(ngf, kernel_size=7,
                strides=1,
                # activation='relu',
                padding='valid',
                kernel_initializer=RandomNormal(0, 0.02),
                bias_initializer='zeros',
                input_shape=INPUT_SHAPE,
                name="encoder_"+name+"_0" )(x)


        # x = BatchNormalization(axis=3,epsilon=1e-5, momentum=0.9, gamma_initializer=RandomNormal(1., 0.02))(x)
        x = InstanceNormalization(axis=3,epsilon=1e-5,  gamma_initializer=RandomNormal(1., 0.02))(x)
        #x = LeakyReLU(0.2)(x)
        x = Activation("relu")(x)

        x = Conv2D(ngf*2, kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=RandomNormal(0, 0.02),
                bias_initializer='zeros',
                name="encoder_"+name+"_1" )(x)
        # x = BatchNormalization(axis=3,epsilon=1e-5, momentum=0.9, gamma_initializer=RandomNormal(1., 0.02))(x)
        x = InstanceNormalization(axis=3,epsilon=1e-5,  gamma_initializer=RandomNormal(1., 0.02))(x)
        #x = LeakyReLU(0.2)(x)
        x = Activation("relu")(x)
        # output shape = (128, 128, 128)

        # g.add(ReflectionPadding2D())

        x = Conv2D(ngf*4, kernel_size=3,
                strides=2,
                padding="same",
                kernel_initializer=RandomNormal(0, 0.02),
                bias_initializer='zeros',
                name="encoder_"+name+"_2",
                )(x)

        # x = BatchNormalization(axis=3,epsilon=1e-5, momentum=0.9, gamma_initializer=RandomNormal(1., 0.02))(x)
        x = InstanceNormalization(axis=3,epsilon=1e-5,  gamma_initializer=RandomNormal(1., 0.02))(x)
        #x = LeakyReLU(0.2)(x)
        # x = Activation("relu")(x)
        # output shape = (64, 64, 256)

        # # END ENCODER



        # # TRANSFORM

        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)

        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)

        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)
        x = resnet_block(x, 64 * 4)


        # # END TRANSFORM
        # # generator.shape = (64, 64, 256)

        # # DECODER with conv2d transpose
        # x = Conv2DTranspose(ngf*2,kernel_size=3, strides=2, padding="same")(x)
        # x = BatchNormalization(axis=3,epsilon=1e-5, momentum=0.9, gamma_initializer=RandomNormal(1., 0.02))(x)
        # x = ReLU()(x)

        # x = Conv2DTranspose(ngf,kernel_size=3, strides=2, padding="same")(x)
        # x = BatchNormalization(axis=3,epsilon=1e-5, momentum=0.9, gamma_initializer=RandomNormal(1., 0.02))(x)
        # x = ReLU()(x)

        # # DECODER with upscale

        x= scaleup(x, ngf * 2)
        x= scaleup(x, ngf)

        x = ReflectionPadding2D(padding=(3,3))(x)
        x = Conv2D(3, kernel_size=7, strides=1, padding="valid", name=name+"_out_layer")(x)
        x = Activation('tanh')(x)
        # exit()
        # END DECODER


    composed = Model(model_input, x, name=name)
    composed.summary()

    return composed, model_input, x


def fromMinusOneToOne(x):
    return x/127.5 -1

def toRGB(x):
    return (1+x) * 127.5


def createImageGenerator( subset="train", data_type="A", batch_size=1, pp=None):

    # we create two instances with the same arguments
    data_gen_args = dict(
                         # rescale = 1./127.5,
                         # rotation_range=5.,
                         preprocessing_function= pp,
                         horizontal_flip=True,
                         # width_shift_range=0.1,
                         # height_shift_range=0.1,
                         # zoom_range=0.1
                         )

    image_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_directory=subset+data_type
    print('data/'+DATASET+'/'+image_directory)
    image_generator = image_datagen.flow_from_directory(
        'data/'+DATASET+'/'+image_directory,
        class_mode=None,
        batch_size=batch_size,
        seed=seed)

    return image_generator

# generate noisy labels and randomly flip them:
# @see https://github.com/soumith/ganhacks
# @see https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9

def generate_discriminator_labels(generator_trainer, margin=0.1, invert_labels_percentage=4):
    ones = np.ones((BATCH_SIZE,)+ generator_trainer.output_shape[0][1:])
    zeros = np.zeros((BATCH_SIZE,)+ generator_trainer.output_shape[0][1:])

    margin = random.uniform(0, margin)
    ones = np.sum([ones, -margin])
    zeros = np.sum([zeros, margin])

    # we randomly invert labels for a small percentage of samples
    x = random.randint(1,100)
    if x < invert_labels_percentage:
        return ones, zeros
    else:
        return zeros, ones

def data_generator(train_A_image_generator, train_B_image_generator, generator_AtoB, generator_BtoA, batch_size=1):

    return


def fit(
    generator_trainer,
    disc_trainer,
    generator_AtoB,
    generator_BtoA
    ):

    fake_A_pool = []
    fake_B_pool = []

    if not IMAGE_RANGE_0_255:
        train_A_image_generator = createImageGenerator("train", "A", pp=fromMinusOneToOne)
        train_B_image_generator = createImageGenerator("train", "B", pp=fromMinusOneToOne)
    else:
        train_A_image_generator = createImageGenerator("train", "A")
        train_B_image_generator = createImageGenerator("train", "B")

    # print(train_A_image_generator.next())
    # for c in train_A_image_generator:
    #     print(c)
    #     exit()
    # exit()


    now = time.strftime("%Y-%m-%d_%H.%M.%S")
    it = 1

    session = K.get_session()
    fw = tf.summary.FileWriter(logdir="./tensorboard/"+now, graph=session.graph, flush_secs=5)

    real_zeros = np.zeros((BATCH_SIZE,)+ generator_trainer.output_shape[0][1:])


    while it  <= ITERATIONS:



        noisy_zeros, noisy_ones = generate_discriminator_labels(generator_trainer)

        start = time.time()
        print("\nIteration %d " % it)
        # simple man solution to decrease LR in generator: every X iterations we reduce the LR by 10%
        if it>1 and not it % 2000:
            K.set_value(generator_trainer.optimizer.lr, K.get_value(generator_trainer.optimizer.lr) * 0.9)
            K.set_value(disc_trainer.optimizer.lr, K.get_value(disc_trainer.optimizer.lr) * 0.9)

            print("IT %d | LR-- | New LR: G: %s D: %s \n " % (it, K.get_value(generator_trainer.optimizer.lr), K.get_value(disc_trainer.optimizer.lr)))


        sys.stdout.flush()

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

        disc_trainer.trainable = True

        # print(K.get_value(disc_trainer.optimizer.lr))

        # K.set_value(disc_trainer.optimizer.lr, K.get_value(disc_trainer.optimizer.lr) * 4)



        for x in range(0, DISCRIMINATOR_ITERATIONS):
            _, D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B = \
            disc_trainer.train_on_batch(
                [real_A, fake_A, real_B, fake_B],
                [ noisy_zeros, noisy_ones , noisy_zeros, noisy_ones] )

        # disc_trainer = Model([real_A, generated_A, real_B, generated_B],
        #              [  discriminator_real_A,
        #                 discriminator_generated_A,
        #                 discriminator_real_B,
        #                 discriminator_generated_B] , name="discriminator_trainer")

        disc_trainer.trainable = False
        print("=====")
        print("Discriminator loss:")
        print("Real A: %s, Fake A: %s || Real B: %s, Fake B: %s " % ( D_loss_real_A, D_loss_fake_A, D_loss_real_B, D_loss_fake_B))



        # on generator training we don't use noisy labels for target
        if USE_IDENTITY_LOSS:
            _, G_loss_fake_B, G_loss_fake_A, G_loss_rec_A, G_loss_rec_B, G_loss_id_A, G_loss_id_B = \
                generator_trainer.train_on_batch(
                    [real_A, real_B],
                    [real_zeros, real_zeros, real_A, real_B, real_A, real_B])
        else:
            _, G_loss_fake_B, G_loss_fake_A, G_loss_rec_A, G_loss_rec_B = \
                generator_trainer.train_on_batch(
                    [real_A, real_B],
                    [real_zeros, real_zeros, real_A, real_B])

                # generator_trainer outputs:
                # [discriminator_generated_B,   discriminator_generated_A,cyc_A,      cyc_B,]





        print("=====")
        print("Generator loss:")

        if USE_IDENTITY_LOSS:
            print("Fake B: %s, Cyclic A: %s || Fake A: %s, Cyclic B: %s || ID A: %s, ID B: %s" % (G_loss_fake_B, G_loss_rec_A, G_loss_fake_A, G_loss_rec_B, G_loss_id_A, G_loss_id_B))
        else:
            print("Fake B: %s, Cyclic A: %s || Fake A: %s, Cyclic B: %s " % (G_loss_fake_B, G_loss_rec_A, G_loss_fake_A, G_loss_rec_B))

        end = time.time()
        print("Iteration time: %s s" % (end-start))
        sys.stdout.flush()

        summary_values = [
            tf.Summary.Value(tag="disc_A/loss_on_real", simple_value = D_loss_real_A),
            tf.Summary.Value(tag="disc_A/loss_on_generated", simple_value = D_loss_fake_A),
            tf.Summary.Value(tag="disc_B/loss_on_real", simple_value = D_loss_real_B),
            tf.Summary.Value(tag="disc_B/loss_on_generated", simple_value = D_loss_fake_B),

            tf.Summary.Value(tag="gen/disc_loss_on_generated_B", simple_value = G_loss_fake_B),
            tf.Summary.Value(tag="gen/disc_loss_on_generated_A", simple_value = G_loss_fake_A),
            tf.Summary.Value(tag="gen/cyc_A", simple_value = G_loss_rec_A),
            tf.Summary.Value(tag="gen/cyc_B", simple_value = G_loss_rec_B),
            tf.Summary.Value(tag="gen/LR", simple_value = K.get_value(generator_trainer.optimizer.lr)),
        ]

        # generator_trainer.get_layer("gen_A").summary()
        # exit()
        # print(generator_trainer.get_layer("gen_A").get_layer('encoder_gen_A_0').output)

        # if not (it % 10) :
        #     generator_trainer.get_layer("gen_A").summary()
        #     o = generator_trainer.get_layer("gen_A").get_layer('encoder_gen_A_0').output


        # fw.flush()

        if not (it % SAVE_IMAGES_INTERVAL ):
            imgA = real_A
            # print(imgA.shape)
            imga2b = generator_AtoB.predict(imgA)
            # print(imga2b.shape)
            imga2b2a = generator_BtoA.predict(imga2b)
            # print(imga2b2a.shape)
            imgB = real_B
            imgb2a = generator_BtoA.predict(imgB)
            imgb2a2b = generator_AtoB.predict(imgb2a)



            if not IMAGE_RANGE_0_255:
                c = np.concatenate([toRGB(imgA), toRGB(imga2b), toRGB(imga2b2a), toRGB(imgB), toRGB(imgb2a), toRGB(imgb2a2b)], axis=2).astype(np.uint8) # IF IMAGE RANGE -1 to 1
            else:
                c = np.concatenate([imgA, imga2b, imga2b2a, imgB, imgb2a, imgb2a2b], axis=2).astype(np.uint8)

            x = Image.fromarray(c[0])


            x.save("data/generated/iteration_%s.png" % str(it).zfill(4), "PNG")

            with open("data/generated/iteration_%s.png" % str(it).zfill(4), "rb") as image_file:
                s = image_file.read()

            i = tf.Summary.Image(encoded_image_string=s, height=c.shape[0], width=c.shape[0])
            summary_values.append(tf.Summary.Value(tag="output/%s" % it, image=i))


        summary = tf.Summary(value=summary_values)

        fw.add_summary(summary, global_step=it)



        # with open("models/generator_AtoB.pickle", "wb") as saveFile:
        #     pickle.dump(generator_AtoB, saveFile)

        # with open("models/generator_BtoA.pickle", "wb") as saveFile:
        #     pickle.dump(generator_BtoA, saveFile)

        if not (it % SAVE_MODEL_INTERVAL):
            generator_AtoB.save("models/generator_AtoB_id.h5")
            generator_BtoA.save("models/generator_BtoA_id.h5")

        it+=1
    fw.close()



    generator_AtoB.save("models/generator_AtoB_id.h5")
    generator_BtoA.save("models/generator_BtoA_id.h5")


    return

if __name__ == '__main__':


    generator_AtoB, input_A, generated_B = generator(INPUT_SHAPE, name="gen_AToB")

    generator_BtoA, input_B, generated_A = generator(INPUT_SHAPE, name="gen_BToA")

    discriminator_A = discriminator(INPUT_SHAPE, name="disc_A")
    discriminator_B = discriminator(INPUT_SHAPE, name="disc_B")



    ### GENERATOR TRAINING
    optim = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    # input_A = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="gt_input_real_A")
    # generated_B = generator_AtoB(input_A)
    discriminator_generated_B = discriminator_B(generated_B)
    cyc_A = generator_BtoA(generated_B)


    # input_B = Input(batch_shape=(None, IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH), name="gt_input_real_B")
    # generated_A = generator_BtoA(input_B)
    discriminator_generated_A = discriminator_A(generated_A )
    cyc_B = generator_AtoB(generated_A)


    # cyclic error is increased, because it's more important
    cyclic_weight_multiplier = 15


    generator_trainer_inputs = [input_A, input_B]

    if USE_IDENTITY_LOSS:
        generator_trainer_outputs= [
            discriminator_generated_B,  # MSE LOSS * 1
            discriminator_generated_A,  # MSE LOSS * 1
            cyc_A,                      # MAE LOSS * cyclic_weight_multiplier
            cyc_B,                      # MAE LOSS * cyclic_weight_multiplier
            generated_B,                # MAE LOSS * 1
            generated_A,                # MAE LOSS * 1
        ]


        losses =         [ "MSE", "MSE", "MAE",                   "MAE",                    "MAE", "MAE"]
        losses_weights = [ 1,     1,     cyclic_weight_multiplier, cyclic_weight_multiplier,  1,     1    ]
    else:
        generator_trainer_outputs = [
            discriminator_generated_B,  # MSE LOSS * 1
            discriminator_generated_A,  # MSE LOSS * 1
            cyc_A,                      # MAE LOSS * cyclic_weight_multiplier
            cyc_B,                      # MAE LOSS * cyclic_weight_multiplier
        ]

        losses =         [ "MSE", "MSE", "MAE",                   "MAE"]
        losses_weights = [ 1,     1,     cyclic_weight_multiplier, cyclic_weight_multiplier]


    generator_trainer =  Model(
        generator_trainer_inputs,
        generator_trainer_outputs,
        name="generator_trainer"
    )
    generator_trainer.compile(optimizer=optim, loss = losses, loss_weights=losses_weights)



    ### DISCRIMINATOR TRAINING

    disc_optim = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    # disc_optim = keras.optimizers.SGD(lr=0.00001, decay=1e-07, momentum=0.9)

    real_A = Input(shape=INPUT_SHAPE, name="dt_input_real_A")
    real_B = Input(shape=INPUT_SHAPE, name="dt_input_real_B")

    generated_A = Input(shape=INPUT_SHAPE, name="dt_input_generated_A")
    generated_B = Input(shape=INPUT_SHAPE, name="dt_input_generated_B")

    discriminator_real_A = discriminator_A(real_A)
    discriminator_generated_A = discriminator_A(generated_A)
    discriminator_real_B =  discriminator_B(real_B)
    discriminator_generated_B = discriminator_B(generated_B)

    disc_trainer = Model([real_A, generated_A, real_B, generated_B],
                         [  discriminator_real_A,
                            discriminator_generated_A,
                            discriminator_real_B,
                            discriminator_generated_B] , name="discriminator_trainer")


    disc_trainer.compile(optimizer=disc_optim, loss = 'MSE')


    #########
    ##
    ## TRAINING
    ##
    #########



    fit(generator_trainer,
        disc_trainer,
        generator_AtoB,
        generator_BtoA)



        # with open("models/generator_AtoB.json", "w") as saveFile:
        #     saveFile.write(generator_AtoB.to_json())
        # generator_AtoB.save_weights("models/generator_AtoB_weights.h5")

        # with open("models/generator_BtoA.json", "w") as saveFile:
        #     saveFile.write(generator_BtoA.to_json())
        # generator_BtoA.save_weights("models/generator_BtoA_weights.h5")




     # lmbd = 10.0,
     #    idloss=1.0,

     #    # optimizers
     #    lr = 0.0002,            ## initial learning rate for adam
     #    beta1 = 0.5,            ## momentum term of
