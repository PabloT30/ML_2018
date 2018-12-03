import tensorflow as tf     # Importing Tensorflow.
import numpy as np          # To treat the data as an python array
import keras
from keras.models import load_model
from keras import backend as K # Library for killing older running programs
from keras.preprocessing import image
import matplotlib.pyplot as plt # For image plotting.
from PIL import Image

K.clear_session() # Killing older running keras applications

# Loading the model

model = load_model('my_model.h5')

# Loading the target image

length, height = 250, 250
img_path = 'data/elbow_test_dataset/negative/2512.png'

# Image with negative result(1):
    # 'data/elbow_test_dataset/negative/2512.png'
# Image with positive result(0):
    # 'data/elbow_test_dataset/positive/2272.png'

img = image.load_img(
    path = img_path, 
    target_size=(250, 250), 
    color_mode='grayscale'
)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

print(x.shape)

plt.imshow(img)
plt.axis('off')
plt.show()

pred = model.predict(x)

print(pred.argmax())

K.clear_session() # Killing older running keras applications