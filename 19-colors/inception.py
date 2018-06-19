import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import sys
import random
import tensorflow as tf

from bcolors import bcolors

from model import restoreModel

TEST_DIR = "data/raf_testing/"
TRAIN_DIR ="data/raf_training/"

BATCH_SIZE = 10
EPOCHS = 1
OPTIMIZER = 'rmsprop'

params = sys.argv

model_to_restore = False
if(len(params) >=2):
    model_to_restore = params[1]


# Get images
X = []
for filename in os.listdir(TRAIN_DIR):
    X.append(img_to_array(load_img(TRAIN_DIR+filename)))
X = np.array(X, dtype=float)


# SCALE IMAGE IN RANGE 0-1
Xtrain = 1.0/255*X


#Load INCEPTION weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()


def buildModel():

    embed_input = Input(shape=(1000,))

    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    # the final model has 2 inputs and one output
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    return model



# take a grayscaled rgb as input
# resize it
# create embeddings

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


# Image transformer
# image generator
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)



# create a batch_size batch of images with the generator
# for each one 
#   create an rgb version 
#   create its embeddings via inception
# for each one
#   create its lab version
#   since its range is -127/+128, resize it to -1/1 with [...] / 128

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)




if model_to_restore:
    model = restoreModel(model_to_restore)
else:
    model = buildModel()

# compile and fit
model.compile(optimizer=OPTIMIZER, loss='mse')
model.fit_generator(image_a_b_gen(BATCH_SIZE), epochs=EPOCHS, steps_per_epoch=1)




# test
color_me = []
for filename in os.listdir(TEST_DIR):
    color_me.append(img_to_array(load_img(TEST_DIR+filename)))
color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))



model_name ="model"

model_config_path = "models/"+model_name+"_config.txt" 
with open(model_config_path, "w") as f:
    print(bcolors.OKGREEN+"Writing model config to: "+model_config_path+"!"+bcolors.ENDC)

    f.write(str(model.get_config()))
"""
with open("models/"+model_name+".txt","w") as f:
    f.write("Optimizer: {0}\nEpochs: {1}\nBatch Size:{2}\n".format(OPTIMIZER, EPOCHS, BATCH_SIZE))
    f.write("validation loss: "+str(score[0])+"\n")
    f.write("validation accuracy: "+str(score[1])+"\n")
    f.write("---\n")
    f.write("Model configuration saved to:\n")
    f.write(model_config_path+"\n")
    f.write("Use model = Model.from_config(configuration_dictionary) to re-create the model")
"""



try:
    model.save("models/"+model_name)
    print(bcolors.OKGREEN+"Model saved to: models/"+model_name+"!"+bcolors.ENDC)

except IOError as e:
    model.save(model_name)
