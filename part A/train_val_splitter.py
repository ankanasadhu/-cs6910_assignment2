import os
import random
import itertools
import cv2
import numpy as np
from PIL import Image
import pickle

class splitter:
    @staticmethod
    def split(path):
        no_of_classes = 10
        train_split = []
        val_split = []
        all_files, total_count = splitter.retrive_names(path)
        no_we_want = int(0.1 * total_count * (1 / no_of_classes))
        for x in all_files:
            val_split.append(x[:no_we_want])
            train_split.append(x[no_we_want : ])
        val_split = list(itertools.chain.from_iterable(val_split))
        train_split = list(itertools.chain.from_iterable(train_split))
        random.shuffle(val_split)
        random.shuffle(train_split)
        return val_split, train_split
    
    @staticmethod
    def retrive_names(path):
        folder_path = path
        total_count = 0
        all_files = []
        for _, sub_folder in enumerate(os.listdir(folder_path)):
            sub_folder_path = folder_path + '/' + sub_folder
            sub_ = []
            for _, filename in enumerate(os.listdir(sub_folder_path)):
                total_count += 1
                sub_.append(sub_folder_path + '/' + filename)
            all_files.append(sub_)
        # print(total_count)
        return all_files, total_count
    
    @staticmethod
    def make_data_set(path_array, image_size):
        image_array = []
        class_name = []
        for path in path_array:
            image= np.array(Image.open(path))
            image= np.resize(image,(image_size,image_size,3))
            image = image.astype('float32')
            image /= 255
            image_array.append(image)
            class_name.append(path.split('/')[3])
        target_dict= {k: v for v, k in enumerate(np.unique(class_name))}
        target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]
        return image_array, target_val

path = 'nature_12K/inaturalist_12K/train'
val_split , train_split = splitter.split(path)
train_images, train_labels = splitter.make_data_set(train_split, 200)
val_images, val_labels = splitter.make_data_set(val_split, 200)

train_data = np.array(train_images)
val_data = np.array(val_images)
train_label = np.array(train_labels)
val_label = np.array(val_labels)

np.save('train_data', train_data)
np.save('val_data', val_data)
# np.save('train_labels', train_label)
# np.save('val_labels', val_label)


new_val_labels = []
new_train_labels = []
for x in val_labels:
    label = [0] * 10
    label[x] = 1
    new_val_labels.append(label)
for x in train_labels:
    label = [0] * 10
    label[x] = 1
    new_train_labels.append(label)

train_label = np.array(new_train_labels)
val_label = np.array(new_val_labels)
np.save('train_label', new_train_labels)
np.save('val_label', new_val_labels)

# val_labels = np.load('val_label.npy')
# train_labels = np.load('train_label.npy')
# train_data = np.load('train_data.npy')
print(train_data.shape)
print(val_data.shape)
print(train_label.shape)
print(val_label.shape)
print(val_labels)
