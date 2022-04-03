# Assignment 2: PART A:

* PART A of the second assignment consists of a CNN trained from scratch. The model was created such that the filter size and the number of layers in the model can easily be changed.
* The above model was subjected to a dataset that has been augmented to prevent overfitting of the model.
* The best model from the above model sweeps was saved and the backpropagation was replaced with guided backpropagation.

# Library used:
* Tensorflow was used to train the CNN model from scratch.
* Keras was used for image preprocessing: data resizing, augmentation, etc.
* PIL library was used for handling of the image dataset.
* numpy was used for the mathematical operations and array functions.
* os library was used to interface with operating system and obtain the folders from the directory.
* mpl_toolkits.axisartist.axislines and matplotlib are used to plot the results of guided backpropagation.
* wandb was used to find the best hyperparameter configuratin and to make insightful observations from the plots obtained.

# How to use?
* **q1.py**
    * This file is created for the first question where all the hyperparameters can be passed through the commandline.
    * To run this code, the following example command can be run on the terminal:

    **python q1.py 0.0004942 128 200,200,3 16,32,64,128,256 2,2,2,2,2 r,r,r,r,r**

    * The first parameter is the learning rate, the scond parameter specifies the number of neurons in the dense layer, the third specifies the image size for input, the fourth parameters are the number of filters in each layer, the fifth parameter specifies the filter sizes in each layer and the last parameter specifies the activation functions used in each layer.
    * **r** - relu, **s** - sigmoid, **t** - tanh
* **train.py**:
    
   - The below variables can be changed in the given file by the user to obtain a model of the required specifications:
    * fltr_size: It is an integer. It consists the filter size.
    * filter_org : It is a float value that increases the number of filters in each subsequent layers by 2 or halves the number of filters.
    * dropout : It is a float.The rate at which inputs are randomly set to zero.
    * batch_normalization : If this value is set to 1, batch normalization layers are added after the convolution layers otherwise it is not added.
    * activations : It is a string. Indicates the type of activation function used by the CNN which by default is set to relu.

    >training_data and validation data consists of the training dataset and validation dataset respectively.
* **guided_backpropagation.py** file stores random neurons in array **neuron_no_arr**. The training dataset is searched one by one and the first image that is excited by each of these neurons due to guided backpropagation is plotted. 

    * **np.sum(gradient)** is checked if it is greater than zero. To check if the numpy array **gradient** (that contains the gradients of the weights from guided back propagation) has atleast one element greater than zero to verify if the image was indeed excited by a certain neuron.

# How is the model trained?
* In train.py file the model created and stored in variable **model** is trained by the augmented dataset created by the data_augmentation.py file.
* The hyperparameters are obtained from different sweeps by wandb and different designs of CNN models are trained.
* RandomCrop adds an additional layer of augmentation as the image dataset is randomly cropped to 200X200 size.
* Successively the convolutional neural networks are created where the total number of layers is variable. With these layers maxpooling and batchnormalization are done.
* Following the above layers a dropout layer is added.
* A dense layer is created at the end which contains 10 neurons to classify the input into correct classes.
The model is trained using:

    model.fit( training_data,epochs=int(config.epochs), validation_data=validation_data,shuffle=True, callbacks=[WandbCallback()])

# Acknowledgement

* The entire assignment has been developed from the lecture videos and slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
*  https://wandb.ai
*  Data Augmentation: https://www.tensorflow.org/tutorials/images/data_augmentation
* Guided Backpropagation: https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/

    https://stackoverflow.com/questions/55924331/how-to-apply-guided-backprop-in-tensorflow-2-0

