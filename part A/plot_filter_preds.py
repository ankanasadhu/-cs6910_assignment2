from matplotlib import pyplot as plt
from tensorflow import keras
import numpy as np
import os
import random
from PIL import Image
from tensorflow.keras.models import Model 

folder_path = 'nature_12K/inaturalist_12K/val/Amphibia/'
random_image = folder_path + random.choice(os.listdir(folder_path))
random_image_data = np.array([np.array(Image.open(random_image).resize((200, 200)))])
loaded_model = keras.models.load_model('best_model')

first_layer = Model(inputs= loaded_model.input, outputs=loaded_model.layers[1].output)
first_layer_op = first_layer.predict(random_image_data)

outputs = []
for x in range(16):
    outputs.append(first_layer_op[0][:,:,x])
plt.figure(figsize=(4,4))
for x in range(16):
    ax = plt.subplot(4, 4, x + 1, xticks=[], yticks=[])
    plt.subplots_adjust(hspace=1, wspace=1)
    plt.axis("off")
    plt.imshow(outputs[x])

plt.savefig('image2.jpg')
