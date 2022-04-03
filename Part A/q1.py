import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import sys


# all the functions used to build, train and validate the model is same as that used in train.py
# the only difference is that we pass the hyperparameter values as arguments to this make_model function
def make_model(learning_rate_=0.0005, dense_=128, image_size_=(200,200,3), no_filters_=[16, 32, 64, 128, 256], fltr_sizes_=[2,2,2,2,2], activations_=['relu','relu','relu','relu','relu']):
    epochs_=10
    build_size = (None, image_size_[0], image_size_[1], 3)
    model = Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomCrop(height=image_size_[0], width=image_size_[1]))
    model.add(Conv2D(no_filters_[0], (fltr_sizes_[0], fltr_sizes_[0]), activation=activations_[0], input_shape=image_size_))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(no_filters_[1], (fltr_sizes_[1],fltr_sizes_[1]), activation=activations_[1]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(no_filters_[2], (fltr_sizes_[2],fltr_sizes_[2]), activation=activations_[2]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(no_filters_[3], (fltr_sizes_[3],fltr_sizes_[3]), activation=activations_[3]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(no_filters_[4], (fltr_sizes_[4],fltr_sizes_[4]), activation=activations_[4]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(dense_, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.build(build_size)
    model.summary()
    model.compile(optimizer = Adam(learning_rate=learning_rate_), loss=CategoricalCrossentropy(), metrics=['acc'])
    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    target_size_ = (400, 400)
    training_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=target_size_, batch_size=32, class_mode='categorical', subset='training')
    validation_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=target_size_, batch_size=32, class_mode='categorical', subset='validation')
    model.fit(
        training_data,
        epochs=epochs_,
        validation_data=validation_data,
        shuffle=True,
        )
    # saving the best model so that it can be used for predictions using the test data
    model.save("best_model")

n = len(sys.argv)
print(n)
acts = {'r' : 'relu', 's' : 'sigmoid', 't' : 'tanh'}

# we take the input values as command line arguments and recieve them using sys.argv
leaning_rate_ = float(sys.argv[1])
dense_ = int(sys.argv[2])
image_size_ = tuple([int(i) for i in sys.argv[3].split(',')])
no_filters_ = [int(i) for i in sys.argv[4].split(',')]
fltr_sizes_ = [int(i) for i in sys.argv[5].split(',')]
activations_ = [acts[i] for i in sys.argv[6].split(',')]

make_model(leaning_rate_, dense_, image_size_, no_filters_, fltr_sizes_, activations_)

# python q1.py 0.0004942 128 200,200,3 16,32,64,128,256 2,2,2,2,2 r,r,r,r,r
