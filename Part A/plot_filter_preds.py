from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import os
import random
from PIL import Image
from tensorflow.keras.models import Model 

# specifying the folder path
folder_path = 'nature_12K/inaturalist_12K/val/Amphibia/'
# a random image is chosen from the directory
random_image = folder_path + random.choice(os.listdir(folder_path))
# the data for the random image is received using the image location
random_image_data = np.array([np.array(Image.open(random_image).resize((200, 200)))])
# load the saved best model
loaded_model = keras.models.load_model('best_model')

# we want the output of the model to be the first feature tensor produced by the image and filters on the first layer
first_layer = Model(inputs= loaded_model.input, outputs=loaded_model.layers[1].output)
# the output produced is the output from the first convolutional layer
first_layer_op = first_layer.predict(random_image_data)

outputs = []
# Since there are 16 filters in the first layer, each filter produces a single feature matrix
# each feature matrix for each filter is stored
for x in range(16):
    outputs.append(first_layer_op[0][:,:,x])
plt.figure(figsize=(4,4))
# the feature matrix are plotted 
for x in range(16):
    ax = plt.subplot(4, 4, x + 1, xticks=[], yticks=[])
    plt.subplots_adjust(hspace=1, wspace=1)
    plt.axis("off")
    plt.imshow(outputs[x])

plt.savefig('image2.jpg')
