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

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # For image preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras import optimizers
from keras import backend as K # Library for killing older running programs

import matplotlib.pyplot as plt # For image plotting.

K.clear_session() # Kill older running keras applications

# Importing and preprocessing the data
    # Data - Total of 2000 Images (1850 for training, 150 for testing)
    # Two classes (binary classification) - Bone broked or not

training_path = './elbow_dataset'
test_path = './elbow_test_dataset'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory = training_path,
    class_mode = 'binary',
    target_size = (256,256)
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    class_mode = 'binary',
    target_size = (256,256)
)

# Creating a sequential model and adding the layers
model = Sequential()

    # Defining the model
        # Feature extractor
            # Convolution
            # Subsampling
            # Convolution
            # Subsampling
        # Classification
            # Fully connected network

model.add(Conv2D(28, kernel_size=(3,3), input_shape=(256,256)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


    # Compiling the model
        # Defining the parameters

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics = ['accuracy'])

    # Training the model

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=5,
        validation_data=test_generator,
        validation_steps=800)

    # Validating the model


# Evaluating the model

model.evaluate(test_generator)

# Saving the model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# Predicting new inputs