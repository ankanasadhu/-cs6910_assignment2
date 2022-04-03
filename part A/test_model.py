from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

loaded_model = keras.models.load_model('best_model')
image_datagen = ImageDataGenerator(rescale=1./255)
test_data = image_datagen.flow_from_directory('nature_12K/inaturalist_12K/val/', shuffle=True, target_size=(400, 400), batch_size=32, class_mode='categorical')

score, acc = loaded_model.evaluate(test_data)
print("accuracy {}".format(str(acc)))

