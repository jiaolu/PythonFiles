# Load pickled data
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D


with open('test.p', mode='rb') as f:
    data = pickle.load(f)

X_train, y_train= data['features'], data['labels']

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D

# TODO: Build Convolutional Neural Network in Keras Here
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1,1), padding='same', input_shape=(32,32,3)))

model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))

# Preprocess data
X_normalized = np.array(X_train / 255.0 - 0.5 )

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, nb_epoch=3, validation_split=0.2)