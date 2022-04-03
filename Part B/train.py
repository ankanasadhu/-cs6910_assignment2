import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback

def train():
    #Setting up WandB
    wandb.init(project="cnn")
    config = wandb.config
    wandb.run.name = "model_name_" + str(config.model_name) + "isMoreDense"+str(config.more_dense)+"trainable_last_no_of_layers_"+str(config.starting_layer)+"_epoch_"+ str(int(config.epochs))
    #Storing the model name 
    model_name= config.model_name
    # Using imagenet weights 
    if model_name== "resnet50":
        pre_trained_model= tf.keras.applications.resnet_v2.ResNet152V2(include_top=False, weights="imagenet", input_shape=(200,200, 3), classes=1000,classifier_activation="softmax",)
    if model_name== "inceptionv3":
        pre_trained_model= tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(200, 200, 3), classes=1000,classifier_activation="softmax",)
    if model_name== "inceptionresnetv2":
        pre_trained_model= tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(200, 200, 3), classes=1000,classifier_activation="softmax",)
    if model_name== "xception":
        pre_trained_model= tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(200, 200, 3), classes=1000,classifier_activation="softmax",)
    pre_trained_model.trainable = False
    total = len(pre_trained_model.layers)
    starting_layer=int(config.starting_layer)
    for i in range(total-starting_layer,total):
        pre_trained_model.layers[i].trainable = True
    model = Sequential()
    #RandomCrop for addition layer of augmentation
    model.add(tf.keras.layers.experimental.preprocessing.RandomCrop(height=200, width=200))
    #Creating an input layer to the CNN
    model.add(tf.keras.Input(shape=(200, 200, 3)))
    #Adding the chosen pre_trained model to the CNN
    model.add(pre_trained_model)
    model.add(tf.keras.layers.Flatten())
    more_dense= int(config.more_dense)
    if more_dense:
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    #Compiling the model
    model.compile(optimizer = Adam(learning_rate=float(config.learning_rate)), loss=CategoricalCrossentropy(), metrics=['acc']) 

    image_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    #Training and Validation data set split
    training_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=(200, 200), batch_size=100, class_mode='categorical', subset='training')
    validation_data = image_datagen.flow_from_directory('dataaug/', shuffle=True, target_size=(200,200), batch_size=100, class_mode='categorical', subset='validation')
   #Running the sweeps for different model configuration
    model.fit(
        training_data,
        epochs=int(config.epochs),
        validation_data=validation_data,
        shuffle=True,
        callbacks=[WandbCallback()]
        )

    
train()
    
