# Fractured bones detection on X-RAY images using Machine Learning
# Technische Universität München
# Final project for the lecture: Advanced Topics in IC Design
# Winter Semester 2018/19 
# Team members:
    # Lisha
    # Richard
    # Pablo
    # YU

# Libraries
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Importing the data
    # Data - Total of 2000 Images (1850 for training, 150 for testing)
    # Two classes (binary classification) - Bone broked or not

# Preprocessing the data

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

    # Compiling the model
        # Defining the parameters
    
    # Training the model

    # Validating the model

# Evaluating the model

# Saving the model

# Predicting new inputs