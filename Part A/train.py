import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
import numpy as np


# this function is called by wandb sweep and it takes in different values for the hyperparameters
def train():

    # initializing wandb
    wandb.init(project="cnn_inaturalist")
    config = wandb.config

    # specifying the number of filters in each layer using filter_org
    no_filters = str(config.filters)
    for x in range(1, int(config.layers)):
        no_filters += '-' + str(int( (float(config.filter_org)** x) * int(config.filters)))
    
    # setting the run name for wandb runs
    wandb.run.name = "no_layers_" + str(config.layers) + "_fltrs_" + no_filters + "_" + "_fltrsize_" + str(config.filter_size) + "_dropout_" + str(config.dropout) + "_epochs_" + str(config.epochs) + "_batch_norm_" + str(config.batch_normalization) +  "_activation_" + str(config.activations)

    # specifying the input shape for the input image 
    input_shape = (200, 200, 3)
    fltr_size = int(config.filter_size)

    # creating a sequential model for the CNN
    model = Sequential()
    # randomly cropping the input images with a specified size as a prprocessing step
    model.add(tf.keras.layers.RandomCrop(height=input_shape[0], width=input_shape[1]))
    # adding the first convolutional layer in the model
    model.add(Conv2D(int(config.filters), (fltr_size, fltr_size), activation=config.activations, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # adding the number of layers after the first layer apecified by the wandb configuration file
    # this value is kept at 5 but can be changed using the sweep.yaml file
    for x in range(1, int(config.layers)):
        model.add(Conv2D(int((float(config.filter_org)** x) * int(config.filters)), (fltr_size, fltr_size), activation=config.activations))
        # if batch norm value in sweep.yaml is 1, then we add a batch normalization layer after each convolutional layer
        if(int(config.batch_normalization)):
            model.add(BatchNormalization(momentum=0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatteing the current output
    model.add(Flatten())
    # adding a dense layer
    model.add(Dense(int(config.dense), activation=config.activations))
    # adding dropout after the dense layer
    if(float(config.dropout) != 0.0):
        model.add(Dropout(float(config.dropout)))
    model.add(Dense(10, activation='softmax'))
    # comiling the model for training 
    model.compile(optimizer = Adam(learning_rate=float(config.learning_rate)), loss=CategoricalCrossentropy(), metrics=['acc'])

    # creating an image data generator object that will be used to flow image data from a directory 
    # to the model for training and validation.
    # the validation data is splitted from the same directory as the training data using the data generator
    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # using the data generator to create train and validation set from the same directory
    training_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=(400, 400), batch_size=32, class_mode='categorical', subset='training')
    validation_data = image_datagen.flow_from_directory('dataaug/', target_size=(400,400), batch_size=32, class_mode='categorical', subset='validation')

    # fitting the model to the training data and validatiing on the validation data
    model.fit(
        training_data,
        epochs=int(config.epochs),
        validation_data=validation_data,
        shuffle=True,
        callbacks=[WandbCallback()]
        )

train()
    