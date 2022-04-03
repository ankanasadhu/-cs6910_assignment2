import os
import random
from PIL import Image
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

number_of_images = 30
labels = []
images = []
loaded_model = keras.models.load_model('best_model')
folder_path = 'nature_12K/inaturalist_12K/val/'
labels_ = {}

# appending the labels using the folder names for classes
i = 0
for count, sub_folder in enumerate(os.listdir(folder_path)):
    labels.append(sub_folder)
    labels_[sub_folder] = i
    i += 1

# generating 30 random images' path to predict using the model and plot the predictions using matplotlib
for x in range(number_of_images):
    image_path = folder_path + random.choice(labels) + '/'
    random_image = random.choice(os.listdir(image_path))
    images.append(image_path + random_image)

# generating and storing the image data for prediction
images_data = []
for x in images:
    data = np.array(Image.open(x).resize((200, 200)))
    images_data.append(data)
images_np_array = np.array(images_data)
# making the predictions using the loaded model
predictions = loaded_model.predict(images_np_array) 
predicted_classes = np.argmax(predictions , axis=1)

parameter = {'axes.titlesize': 6}

plt.figure(figsize=(3, 10))
for x in range(number_of_images):
    plt.rcParams.update(parameter)
    ax = plt.subplot(10, 3, x + 1, xticks=[], yticks=[])
    # if the predicted class is correct, then the title color is green otherwise it is red
    if predicted_classes[x] == labels_[images[x].split('/')[3]]:
        color = 'green'
    else:
        color = 'red'
    ax.set_title(images[x].split('/')[3], color=color)
    plt.subplots_adjust(hspace=1, wspace=1)
    plt.axis("off")
    plt.imshow(images_data[x])

plt.savefig('image.jpg')




