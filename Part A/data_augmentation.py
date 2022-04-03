import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
#Obtaining the files from the directory
folder_path = 'nature_12K/inaturalist_12K/train'
i=0
for count, sub_folder in enumerate(os.listdir(folder_path)):
    sub_folder_path = folder_path + '/' + sub_folder
    dirName='dataaug/' + sub_folder
    os.makedirs(dirName)   
    for count_1, filename in enumerate(os.listdir(sub_folder_path)):
        file_name = f"{sub_folder_path}/{filename}"
        # Converting the input sample image to an array
        im = Image.open(file_name)
        # Reshaping the input image
        newsize = (200, 200)
        im= im.resize(newsize)
        x=np.array(im)
        x= x.reshape((1, ) + x.shape)  
        # Generating and saving  augmented samples 
        try:
            datagen = ImageDataGenerator( horizontal_flip = True)
            for j in datagen.flow(x, save_to_dir =dirName, save_prefix =sub_folder.lower()+str(i), save_format ='jpeg'):
                i+=1
                break
            datagen = ImageDataGenerator(rotation_range=40)
            for j in datagen.flow(x, save_to_dir =dirName, save_prefix =sub_folder.lower()+str(i), save_format ='jpeg'):
                i+=1
                break
        except:
            print('exception occured')
            continue
        
      