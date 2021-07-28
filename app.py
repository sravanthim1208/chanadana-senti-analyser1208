import flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import cv2
import string

string.punctuation
import nltk  # Natural Language tool kit

nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords.words('english')
import os
import matplotlib.pyplot as plt
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# define the flask app
app = Flask(__name__)
# Load model
filename = 'svm_model.h5'
model = pickle.load(open(filename, 'rb'))

# Load labels
filename = 'vectorizer.h5'
vectorizer = pickle.load(open(filename, 'rb'))
DEFAULT_IMAGE_SIZE = tuple((256, 256))


def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if
                                    word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


def predict_output(user_input):
    cleaned_message = message_cleaning(user_input)
    cleaned_message = ' '.join(cleaned_message)
    normalised_msg = [cleaned_message]
    vectorised_input = vectorizer.transform(normalised_msg)
    predicted_output = model.predict(vectorised_input)

    return predicted_output[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # get the file from post request
        input_value = str(request.data)

        result = predict_output(input_value)
        object = {};
        final_=int(result)
        text="Negative"
        if(final_==0):
            text="Positive  &#128525; "
        else:
            text="Negative &#128531;"
        object["result"] = text;
        return object
    return None


if __name__ == '__main__':
    app.run(port=5926)

