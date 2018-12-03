import tensorflow as tf     # Importing Tensorflow.
import numpy as np          # To treat the data as an python array
import keras
from keras.models import load_model
from keras import backend as K # Library for killing older running programs
from keras.preprocessing import image as imagenes
import matplotlib.pyplot as plt # For image plotting.
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import *
#from tkFileDialog import askdirectory


window = Tk()

window.title("Fractured Bones Detection")

window.geometry("420x440")

K.clear_session() # Killing older running keras applications

# Loading the model


length, height = 250, 250
#img_path = 'data/elbow_test_dataset/negative/2512.png'

def prediction():
	etiqueta.configure(text="")
	filename = filedialog.askopenfilename(initialdir = "/home/pablot30/Dokumente/TUM_W1819/programs/bones_ML",title = "Select image",filetypes = (("png files","*.png"),("all files","*.*")))
	image2 = imagenes.load_img(
    	path = filename, 
    	target_size=(400, 300), 
    	color_mode='grayscale'
	)
	x_ray_photo = ImageTk.PhotoImage(image2)

	x_ray_label = Label(
		window,
		image = x_ray_photo
	)
	x_ray_label.image = x_ray_photo # keep a reference!
	x_ray_label.grid(column=0,row=0)
	
	img = imagenes.load_img(
    	path = filename, 
    	target_size=(250, 250), 
    	color_mode='grayscale'
	)
	x = imagenes.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x /= 255.

	modelo = load_model(filedialog.askopenfilename(initialdir = "/home/pablot30/Dokumente/TUM_W1819/programs/bones_ML",title = "Select model",filetypes = (("h5 files","*.h5"),("all files","*.*"))))

	pred = modelo.predict(x)
	print(pred.argmax())

	if(pred.argmax()==1):
		etiqueta.configure(text="Positive")
	elif(pred.argmax()==0):
		etiqueta.configure(text="Negative")

	return 0

# Labeling objects

image = imagenes.load_img(
    path = 'none.png', 
    target_size=(400, 300), 
    color_mode='grayscale'
)
none_photo = ImageTk.PhotoImage(image)

none_label = Label(window,
	image = none_photo
)
none_label.image = none_photo # keep a reference!
none_label.grid(column=0,row=0)

button_predict = Button(
	window,
	text="Predict",
	font=("Times New Roman",14),
	command=prediction,
)
button_predict.grid(column=1, row=0)

etiqueta = Label(
	text="",
	font=("Times New Roman",18),
)
etiqueta.grid(column=0,row=1)

# Give the functionality
	# Function to predict
	# Function to load image
	# Function to load model

window.mainloop()

K.clear_session() # Killing older running keras applications