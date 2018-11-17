import tensorflow as tf     # Importing Tensorflow.
import numpy as np          # To treat the data as an python array
import keras
from keras.models import load_model
from keras import backend as K # Library for killing older running programs
import matplotlib.pyplot as plt # For image plotting.

K.clear_session() # Killing older running keras applications

# Loading the model

model = load_model('my_model.h5')

# Loading the target image

# Predicting the fracture
#pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
#print(pred.argmax())


K.clear_session() # Killing older running keras applications