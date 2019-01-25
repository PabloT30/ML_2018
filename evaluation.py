import tensorflow as tf     # Importing Tensorflow.
import numpy as np          # To treat the data as an python array
import keras
from keras.models import load_model
from keras import backend as K # Library for killing older running programs
import matplotlib.pyplot as plt # For image plotting.
from keras.preprocessing.image import ImageDataGenerator

test_path = 'data/elbow_test_dataset'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(256, 256),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical')

model = load_model('my_densemodel.h5')

x,y = test_generator.next()

#   plt.show()

#for name, param in model.parameters():
#    if param.requires_grad:
#        print(name, param.data)

model.summary()       

for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print (g)
        print (h)

# Test the models
score = model.evaluate(x, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

