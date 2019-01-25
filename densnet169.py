from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # For image preprocessing
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import Callback
from keras.History import history
from keras import backend as K # Library for killing older running programs
from keras.optimizers import Adam

# data generator definition
# generate some new images
def getDataGenerator(train_phase,rescale=1./255):
    if train_phase == True:
        datagen = ImageDataGenerator(
        rotation_range=0.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        channel_shift_range=0.,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=rescale)
    else: 
        datagen = ImageDataGenerator(
        rescale=rescale
        )

    return datagen

# definition of single layer
def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x

# definition of dense block
def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

# the definition of dense net
def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(model_input)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                            beta_regularizer=l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose: 
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet

# set up the path
training_path = 'data/elbow_dataset'
test_path = 'data/elbow_test_dataset'

#define DenseNet parms
ROWS = 256
COLS = 256
CHANNELS = 3
nb_classes = 2
batch_size = 32
nb_epoch = 40
img_dim = (ROWS,COLS,CHANNELS)
densenet_depth = 40
densenet_growth_rate = 12

x_train = train_generator.astype('float32')
x_test = train_generator.astype('float32')
x_train /= 255
x_test /= 255

# train the data 
train_datagen = getDataGenerator(train_phase=True)
train_generator = train_datagen.flow_from_directory(
    directory = training_path,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    target_size = (256,256)
)
# ?? the validation here should be seperated 
validation_datagen = getDataGenerator(train_phase=False)
validation_datagen = validation_datagen.flow(x_test,batch_size = batch_size)


# create the model
model = createDenseNet(nb_classes=nb_classes,img_dim=img_dim,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)
if resume == True: 
    model.load_weights(check_point_file)

optimizer = Adam()

#optimizer = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),

model_checkpoint = ModelCheckpoint(check_point_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

#callbacks=[lr_reducer,model_checkpoint]

model.fit_generator(generator=train_datagen,
                    steps_per_epoch= x_train.shape[0], # batch_size,
                    epochs=nb_epoch,
                    callbacks=[model_checkpoint],
                    validation_data=validation_datagen,
                    validation_steps = x_test.shape[0], # batch_size,
                    verbose=1)

# test the model
model = createDenseNet(nb_classes=nb_classes,img_dim=img_dim,depth=densenet_depth,
                  growth_rate = densenet_growth_rate)
    model.load_weights(check_point_file)
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


#test
test_datagen = getDataGenerator(train_phase=False)
test_datagen = test_datagen.flow(x_test,batch_size = batch_size,shuffle=False)

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate_generator(test_datagen,
                                    steps=x_test.shape[0] // batch_size,
                                    workers=4)
print('Model Accuracy = %.2f' % (evaluation[1]))
