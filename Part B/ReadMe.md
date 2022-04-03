# Assignment 2 PART B:
*   The PART B of assignment 2 is an application of pretrained models : inceptionresnetV2, resnet50, xception and inceptionV3.
* The model weights from training dataset of "imagenet" was used on inaturalist dataset.
# Library used:
* Tensorflow was used to create the CNN model from scratch.
* Keras was used for image preprocessing: data resizing, augmentation, etc.
* tf.keras.applications to import and use the various pre trained models: inceptionresnetV2, resnet50, xception and inceptionV3.
* wandb was used to find the best hyperparameter configuratin and to make insightful observations from the plots obtained.

# How to Use?
*   The following variables can be changed to swap pre trained models:
    *   model_name: string value. Name of the pretrained model to be loaded
    *   more_dense: boolean value. It decides if the user wants another layer before the output layer.
    *   starting_layer: int value. The layer from which the pretrained model is unfreezed and trained.

# How is the model trained?
*    **training_data** and **validation_data** are obtained and the model consisting of a Random Crop layer, Input tensor, the pre_trained model, an optional dense layer and the final output dense layer.
*  model.fit(training_data, epochs=int(config.epochs),validation_data=validation_data, shuffle=True, callbacks=[WandbCallback()])

# Acknowledgement:
* The entire assignment has been developed from the lecture videos and slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
*  https://wandb.ai
* Pre Trained Models: https://keras.io/api/applications/