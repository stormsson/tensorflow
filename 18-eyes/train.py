#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import keras
import math
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image
import random

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from bcolors import bcolors
import sys
import tensorflow as tf
params = sys.argv

model_to_restore = False
if(len(params) >=2):
    model_to_restore = params[1]

# training params
EPOCHS = 30

if(len(params)>=3):
    EPOCHS = int(params[2])

OPTIMIZER = "sgd"
BATCH_SIZE = 64

#LOSS_FUNCTION = keras.losses.mean_squared_error
LOSS_FUNCTION = keras.losses.mean_absolute_error
LOSS_FUNCTION = keras.losses.mean_absolute_percentage_error


# optimizer params
OPTIMIZER_LR = 0.00001
OPTIMIZER_DECAY = 1e-7

# images params
RESIZE_FACTOR = 1

IMG_WIDTH = 384
IMG_HEIGHT = 286
#rows, cols, channels
IMG_SIZE = ( IMG_HEIGHT, IMG_WIDTH, 1 )
#IMG_SIZE = ( 143, 192, 1 )

img_db_file = "images_db_"+str(IMG_SIZE[0])+"x"+str(IMG_SIZE[1])+".npy"



def showDebugImage(data, coords):
    try:
        plt.imshow(data, cmap="gray")
        plt.plot([coords[0], coords[2]],[coords[1], coords[3]],'+', color="red")
        plt.show()
    except TypeError as e:
        print(bcolors.FAIL + "Warning, debug image cannot be shown if data has already been reshaped!"+ bcolors.ENDC)


def restoreModel(restore_file):

    if isfile(restore_file):
        model = load_model(restore_file)
        print(bcolors.OKGREEN+"restored model "+restore_file+"!"+bcolors.ENDC)
        return model

def buildModel(input_shape):

    # img_rows , img_cols, channels

    # r_eye X, r_eye Y , l_eye X, l_eye Y
    output_size = 4

    with tf.device('/gpu:0'):

        model = Sequential()
        model.add(Conv2D(48, kernel_size=(11, 11),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3)))


        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))        
        #model.add(Dropout(0.3))

        model.add(Dense(512, activation='relu'))        
        model.add(Dropout(0.25))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))



        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_size, activation='sigmoid'))

    return model

# optimizers:

def getOptimizer():
    # sgd = keras.optimizers.SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True)
    optims = {
        "sgd" :  keras.optimizers.SGD(lr=OPTIMIZER_LR, decay=OPTIMIZER_DECAY, momentum=0.9),
        "adam": keras.optimizers.Adam(lr=OPTIMIZER_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=OPTIMIZER_DECAY),
        "rmsprop": keras.optimizers.RMSprop(lr=OPTIMIZER_LR, rho=0.9, epsilon=1e-08, decay=OPTIMIZER_DECAY),
        "adadelta": keras.optimizers.Adadelta(lr=OPTIMIZER_LR, rho=0.95, epsilon=1e-08, decay=OPTIMIZER_DECAY)
        # "adadelta": keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    }


    return optims[OPTIMIZER]



def loadData(directory):
    x = []
    y = []

    if(isfile(img_db_file)):
        images = np.load(open(img_db_file, "rb"))
        # print(images["x"][0])
        # print(images["y"][0])
        # exit()
        x = images["x"]
        y = images["y"]

        coords = y[0]
        coords[0] = coords[0] * IMG_WIDTH
        coords[1] = coords[1] * IMG_HEIGHT
        coords[2] = coords[2] * IMG_WIDTH
        coords[3] = coords[3] * IMG_HEIGHT
        #print(coords)
        #showDebugImage(np.reshape(x[0],(IMG_HEIGHT, IMG_WIDTH)) , coords)
        #exit()
    else:
        db = pickle.load(open("eyes_db.pickle", "rb"))
        files = [f for f in listdir(directory) if isfile(join(directory, f)) and "jpg" in f]

        cnt = 0
        random.shuffle(files)

        for f in files:
            fileParts = f.split(".")            
            coords = np.array(db[fileParts[0]], dtype="float32")        

            # [ l_eye_x, l_eye_y, r_eye_x, r_eye_y]

            coords = coords / RESIZE_FACTOR
            
            # NORMALIZATION 0->1
            coords[0] = coords[0] / IMG_WIDTH
            coords[1] = coords[1] / IMG_HEIGHT
            coords[2] = coords[2] / IMG_WIDTH
            coords[3] = coords[3] / IMG_HEIGHT

            img = Image.open(directory+"/"+f)
            img.load()

            data = np.asarray(img, dtype="float32")

            # NORMALIZATION 0->1
            data = data / 255

            data = np.reshape(data, IMG_SIZE)

            x.append(data)
            y.append(coords)

            cnt = cnt +1

            # if(cnt > 99):
            #     break
            # print("appending", fileParts[0])
            #print("coords: ",coords)

            #showDebugImage(data, coords)
            #exit()


        x = np.array(x)
        y = np.array(y)

        np.savez(open(img_db_file, "wb"), x=x, y=y)
        print("saved img_db_file!")

    return x,y


def getModelCallbacks():

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=True)
    earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience= 7)
    reduceLROnPlateau_callback = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr= 1e-9)

    return [ tensorboard_callback, earlystop_callback, reduceLROnPlateau_callback ]


x,y = loadData("images/"+str(IMG_SIZE[0]))
train_percentage = .15
train_size = int(math.floor(train_percentage * len(x)))
print("Loaded %d samples. %d will be used for testing" % (len(x), train_size))

x_train = np.array(x[:-train_size])
y_train = np.array(y[:-train_size])

x_validation = np.array(x[-train_size:])
y_validation = np.array(y[-train_size:])

input_shape = IMG_SIZE
# print(x_train[0])
# exit()
print("Training samples: %d , testing samples: %d. Single sample shape: %s " % (len(x_train), len(x_validation), x_train[0].shape) )

if model_to_restore:
    model = restoreModel(model_to_restore)
else:
    model = buildModel(input_shape)


model.compile(loss= LOSS_FUNCTION,
              optimizer=getOptimizer(),
              #metrics=['accuracy']
              metrics=["mae","accuracy"]
              )

# $ > tensorboard --logdir path_to_current_dir/Graph



model.fit(x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    validation_data=(x_validation, y_validation),
    callbacks= getModelCallbacks()
    )

score = model.evaluate(x_validation, y_validation, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

# optimizer, epochs, batch_size
model_name = "model-{0}-{1}ep-{2}batchsize.h5".format(OPTIMIZER, EPOCHS, BATCH_SIZE)
if(model_to_restore):
    model_name = "RESTORED-"+model_name

with open("models/"+model_name+".txt","w") as f:
    f.write("Optimizer: {0}\nEpochs: {1}\nBatch Size:{2}\n".format(OPTIMIZER, EPOCHS, BATCH_SIZE))
    f.write("validation loss: "+str(score[0])+"\n")
    f.write("validation accuracy: "+str(score[1])+"\n")

try:
    model.save("models/"+model_name)
except IOError as e:
    model.save(model_name)
