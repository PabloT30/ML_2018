import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet169
# this part will prevent tensorflow to allocate all the avaliable GPU Memory
# backend
import tensorflow as tf
from keras import backend as k

# Don't pre-allocate memory; allocate as-needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# Hyperparameters
batch_size = 128
num_classes = 2
epochs = 10
l = 10
num_filter = 2

# Load Data 
training_path = 'data/elbow_dataset'
validation_path = 'data/elbow_validation_dataset' 

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(256, 256),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(256, 256),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical')

#X_batch, y_batch = test_datagen.flow(test_path, test_path, batch_size=42)
#test_generator = test_datagen.flow_from_directory(
#        test_path,
#        target_size=(256, 256),
#        batch_size=32,
#        color_mode = 'grayscale',
#        class_mode='categorical')
# Dense Block
def add_denseblock(input):
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(relu)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
        
    return temp

def add_transition(input):
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(num_filter, (1,1), use_bias=False ,padding='same')(relu)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg

def output_layer(input):
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    
    return output

input = Input(shape=(256,256, 1))
First_Conv2D = Conv2D(num_filter, (3,3), use_bias=False ,padding='same',input_shape=(256, 256, 1))(input)

First_Block = add_denseblock(First_Conv2D)
First_Transition = add_transition(First_Block)

Second_Block = add_denseblock(First_Transition)
Second_Transition = add_transition(Second_Block)

Third_Block = add_denseblock(Second_Transition)
Third_Transition = add_transition(Third_Block)

Last_Block = add_denseblock(Third_Transition)
output = output_layer(Last_Block)

model = Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=5)

model.save('my_densemodel_e15_s20.h5')
# Test the model
#score = model.evaluate(X_batch, y_batch, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

