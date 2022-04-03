from tensorflow.keras.preprocessing import image
from keras.models import load_model
from mpl_toolkits.axisartist.axislines import Subplot
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import os


def guidedRelu(input):
  #gradient calculation changes to guided back propagation
  def grad(gradient):
    return tf.cast(gradient>0,"float32") * tf.cast(input>0, "float32") * gradient
  #forward propagation remains unchanged
  return tf.nn.relu(input), grad

def guided_relu_gradient():
    for layer in model.layers:
        # if layer is activated by relu activation the back propagation of layer is changed to guided relu
        if hasattr(layer, 'activation'):
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
    model.compile()


model=load_model('C:\\Users\\DELL\\Desktop\\Deep Learning\\CNN\\My_Model')
folder_path = 'C:\\Users\\DELL\\Desktop\\Deep Learning\\CNN\\dataaug\\'
images=[]
true_image=[]
#array storing 10 random neuron numbers of the 5th convolutional layer
neuron_no_arr=[6, 8, 10 , 14 , 16 , 19, 22, 25, 28 , 30]
for i in range (10):
    
# accessing the images of each class 
    for count, sub_folder in enumerate(os.listdir(folder_path)):
        sub_folder_path = folder_path + '\\' + sub_folder
        
        for count_1, filename in enumerate(os.listdir(sub_folder_path)):
            name = f"{sub_folder_path}\\{filename}"
            loaded_image = image.load_img(name, target_size=(200, 200))
            img_arr = image.img_to_array(loaded_image)
            #expanding the dimension of images from (200,200,3) to (1, 200,200,3) to match the dimensions of the convolutional neural network
            img_numpy = np.expand_dims(img_arr, axis=0)
            #applying guided backpropagation to the fifth convolutional neural network layer
            neuron_output = tf.keras.activations.relu(model.get_layer('conv2d_4').output[:,:,:,neuron_no_arr[i]])
            neuron = tf.keras.models.Model(inputs = [model.inputs],outputs = [neuron_output])
            with tf.GradientTape() as tape:
                # creating a tensor
                img_tensor=tf.cast(img_numpy,tf.float32)
                #tracking the image tensor for differentiation
                tape.watch(img_tensor)
                #for differentiation the output function is being mapped
                output = neuron(img_tensor)
            gradient= tape.gradient(output,img_tensor).numpy()[0]
            #storing the images that are excited by the current neuron under consideration
            if np.sum(gradient)>0 :
                images.append(gradient)
                true_image.append(loaded_image)
                break
        
                    
        break 
    
    
# replace gradients of backpropagation with guided relu activation
guided_relu_gradient()

fig = pyplot.figure(figsize =(15, 15))
ax = Subplot(fig, 111)
fig.add_subplot(ax)
c=1
for index_1 in  images:
    pyplot.subplot(2,5,c) 
    pyplot.title('by neuron:'+ str(neuron_no_arr[c-1]))
    pyplot.imshow(index_1)
    c+=1
# Images that got excited by the respective neurons in order
pyplot.show()
c=1
for index_2 in  true_image:
    pyplot.subplot(2,5, c) 
    pyplot.title("by neuron:"+ str(neuron_no_arr[c-1]))
    pyplot.imshow(index_2)
    c+=1
 
    

# show the figure
pyplot.show()
