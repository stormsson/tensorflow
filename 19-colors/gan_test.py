""" GAN

Usage:
    gan_test.py <model> <path> [-o OUTPUT] [-g] [-c]

Options:
    -o OUTPUT --output=<OUTPUT>      specify output file
    -g                               convert the input image to grayscaled rgb first
    -c --compose                     compose an image
"""

from docopt import docopt

from keras.preprocessing.image import   img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import numpy as np
from os.path import isfile
from os import listdir

from bcolors import bcolors


from utils.saveload import restore_model, save_model
from utils.misc import l1_loss, l2_loss, generate_noise
from utils.rgb2lab import extract_L_channel_from_RGB

from PIL import Image
import math

CHUNK_SIZE = 256

def compose_image(model, inputFile, outputFile, return_data=False):
    img = Image.open(inputFile)
    img_width, img_height = img.size

    h_chunks = int(math.ceil(img_width / CHUNK_SIZE))
    v_chunks = int(math.ceil(img_height / CHUNK_SIZE))

    new_img = Image.new('RGB',(h_chunks * CHUNK_SIZE, v_chunks * CHUNK_SIZE))

    for x in xrange(0, h_chunks):
        for y in xrange(0, v_chunks):

            box = (x * CHUNK_SIZE, y * CHUNK_SIZE, x * CHUNK_SIZE + CHUNK_SIZE, y*CHUNK_SIZE + CHUNK_SIZE)
            chunk = img.crop(box)
            chunk_array = np.array(chunk.getdata())

            chunk_array = chunk_array.reshape((1, CHUNK_SIZE, CHUNK_SIZE, 3))

            chunk_array = chunk_array / 255.0


            noise = generate_noise(1, 256, 3)
            prediction = model.predict({"gen_noise_input": noise, "gen_bw_input": chunk_array })[0]
            # prediction = model.predict([chunk_array])[0]
            prediction  = (prediction * 255).astype(np.uint8)
            generated_img  = Image.fromarray(prediction)

            new_img.paste(generated_img,(x * CHUNK_SIZE, y * CHUNK_SIZE))


    if return_data:
        return np.fromstring(new_img.tobytes(), dtype=np.uint8)
    else:

        new_img.save(outputFile)


if __name__ == '__main__':
    arguments = docopt(__doc__, version="GAN Test 1.0")
    #print(arguments)
    np.set_printoptions(threshold='nan')


    custom_objects = {
        "l1_loss": l1_loss,
        "l2_loss": l2_loss
    }

    model = restore_model(arguments["<model>"], custom_objects)


    input_img = []
    original_imgs = []
    #read input image

    if isfile(arguments["<path>"]):
        if arguments["--compose"]:
            compose_image(model, arguments["<path>"], arguments["--output"])
            exit()

        input_img = img_to_array(load_img(arguments["<path>"]))
        original_imgs.append(input_img)

        #if(arguments["-g"]):
        input_img = gray2rgb(rgb2gray(1.0/255*input_img))

        #input_img = np.array(input_img)
        input_img = input_img.reshape(1, 256, 256, 3)
        input_img = [ input_img ]



    else:
        for filename in listdir(arguments["<path>"]):
            i = img_to_array(load_img(arguments["<path>"]+filename))
            original_imgs.append(i)

            i = gray2rgb(rgb2gray(1.0/255*i))
            i = i.reshape(1, 256, 256, 3)
            input_img.append(i)

        pass






    if arguments["--output"]:
        cnt = 0
        for img in input_img:

            paired_image = Image.new('RGB',(768, 256))

            o =Image.fromarray(original_imgs[cnt].astype(np.uint8))


            print(img[0])
            # if the image has been converted to grayscale it has been converted in range 0-1 instead of 0-255
            bw_image = Image.fromarray((img[0]*255).astype(np.uint8))

            noise = generate_noise(1, 256, 3)
            prediction = model.predict({"gen_noise_input": noise, "gen_bw_input": img })[0]
            prediction  = (prediction * 255).astype(np.uint8)
            generated_img  = Image.fromarray(prediction)

            paired_image.paste(o,(0,0))
            paired_image.paste(bw_image,(256,0))
            paired_image.paste(generated_img,(512,0))


            if len(original_imgs) > 1:
                paired_image.save("tmp/output-%d.jpg" % cnt)
            else:
                paired_image.save(arguments["--output"])

            cnt+=1

    else:
        for img in input_img:
            print(model.predict(img)[0])




