# Fractured bones detection on X-RAY images using Machine Learning 
# (Forearm, elbow or humerus)
# Technische Universität München
# Final project for the lecture: Advanced Topics in IC Design
# Artificial Intelligence for Embedded Systems
# Winter Semester 2018/19 
# Team members:
    # Lisha
    # Richard
    # Pablo
    # YU

# Libraries
import sys  # sys and os are libraries to allow movement throught the pc files and folders.
import os

import tensorflow as tf     # Importing Tensorflow.

import numpy as np          # To treat the data as an python array

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  
from glob import glob

#from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # For image preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras import optimizers
from keras import backend as K # Library for killing older running programs

import matplotlib.pyplot as plt # For image plotting.

K.clear_session() # Kill older running keras applications

# Importing and preprocessing the data
    # Data - Total of 2000 Images (1850 for training, 150 for testing)
    # Two classes (binary classification) - Bone broked or not

epochs = 2
length, height = 250, 250
batch_size = 32
steps = 1000
validation_steps = 300
filtersConv1 = 32
filtersConv2 = 64
size_filter1 = (3, 3)
size_filter2 = (2, 2)
tamano_pool = (2, 2)
classes = 2
lr = 0.0004

training_path = 'data/elbow_dataset'
test_path = 'data/elbow_test_dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    directory = training_path,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    target_size = (length,height)
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    target_size = (length,height)
)

# Creating a sequential model and adding the layers
cnn = Sequential()

    # Defining the model
        # Feature extractor or learning (convolution + RELU, pooling)
            # Convolution
            # Subsampling
            # Convolution
            # Subsampling
        # Classification
            # Fully connected network)

cnn.add(Convolution2D(filtersConv1, size_filter1, padding ="same", input_shape=(length, height, 1), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtersConv2, size_filter2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

    # Compiling the model
        # Defining the parameters

cnn.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(lr=lr),
    metrics=['accuracy']
)

    # Training the model

cnn.fit_generator(
    train_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps
)

    # Validating the model

# Evaluating the model

#cnn.evaluate(test_generator)

# Saving the model

cnn.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
'''
Model saved with an accuracy of 97.07%
'''

# Predicting new inputs
'''
Prediction done in the "predicition.py" file
'''

K.clear_session() # Kill older running keras applications