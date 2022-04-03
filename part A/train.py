# from gc import callbacks
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback
import numpy as np


# train_data = np.load('train_data.npy')
# train_labels = np.load('train_label.npy')
# val_data = np.load('val_data.npy')
# val_labels = np.load('val_label.npy')

def train():

    wandb.init(project="cnn_inaturalist")
    config = wandb.config

    no_filters = str(config.filters)
    for x in range(1, int(config.layers)):
        no_filters += '-' + str(int( (float(config.filter_org)** x) * int(config.filters)))
    
    wandb.run.name = "no_layers_" + str(config.layers) + "_fltrs_" + no_filters + "_" + "_fltrsize_" + str(config.filter_size) + "_dropout_" + str(config.dropout) + "_epochs_" + str(config.epochs) + "_batch_norm_" + str(config.batch_normalization) +  "_activation_" + str(config.activations)

    # "_bs_" + str(config.batch_size) +

    input_shape = (200, 200, 3) # still left to be fully specified
    fltr_size = int(config.filter_size)

    model = Sequential()
    model.add(tf.keras.layers.RandomCrop(height=input_shape[0], width=input_shape[1]))
    model.add(Conv2D(int(config.filters), (fltr_size, fltr_size), activation=config.activations, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for x in range(1, int(config.layers)):
        model.add(Conv2D(int((float(config.filter_org)** x) * int(config.filters)), (fltr_size, fltr_size), activation=config.activations))
        if(int(config.batch_normalization)):
            model.add(BatchNormalization(momentum=0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(int(config.dense), activation=config.activations))
    if(float(config.dropout) != 0.0):
        model.add(Dropout(float(config.dropout)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer = Adam(learning_rate=float(config.learning_rate)), loss=CategoricalCrossentropy(), metrics=['acc']) # CategoricalAccuracy()

    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    training_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=(400, 400), batch_size=32, class_mode='categorical', subset='training')
    validation_data = image_datagen.flow_from_directory('dataaug/', target_size=(400,400), batch_size=32, class_mode='categorical', subset='validation')

    model.fit(
        training_data,
        epochs=int(config.epochs),
        validation_data=validation_data,
        shuffle=True,
        callbacks=[WandbCallback()]
        )

train()
    