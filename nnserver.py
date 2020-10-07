import binascii
import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras_preprocessing.image import save_img
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from keras.initializers import glorot_uniform


import time

import flask

from flask import request
from flask import jsonify
from flask import Flask
from flask import send_file
from flask import send_from_directory

app = Flask(__name__)

def get_model():
    global model
    model = keras.models.load_model('ts_model.h5', compile=False)
    

    print(' * MODEL LOADED')

def preproccess_image(image, target_size):
    # if image.mode != 'RGB':
    #     image = image.convert('RGB')

    image = image.resize(target_size)
    image = img_to_array(image)
    image = (image - 127.5) / 127.5
    image =np.expand_dims(image, axis=0)
    return image    

print(' * Loading Keras model')
get_model()

@app.route('/')
def home():
    return 'Hello World'

@app.route('/generate', methods=['GET', 'POST'])    
def generate():
    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            print(image)

            image = Image.open(io.BytesIO(image))    

            image = preproccess_image(image, target_size=(256, 256))
            preds = model.predict(image)
            epoch_time = int(time.time())
            outputfile = 'output_%s.png' % (epoch_time)
            print(' * SHAPE IS : ' + str(preds.shape))

            output = tf.reshape(preds, [256, 256, 3])
            output = (output + 1) / 2

            save_img(outputfile, img_to_array(output))

            response = {'result' : outputfile}
    return jsonify(response)

@app.route('/download/<fname>', methods=['GET'])
def download(fname):
    return send_file(fname)

