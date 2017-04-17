# Udacity Self-Driving Car Engineer Nanodegree

## Deep Learning 

Behavioral Cloning: Navigating a Car in a Simulator
---

The goal of this project is to train a deep neural network to drive a car in a simulator as a human 
would do. The steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The following animations show the performance of the model on the two tracks

[//]: # (Image References)

[image1]: ./plots/placeholder.png "Model Visualization"
[image2]: ./plots/placeholder.png "Grayscaling"
[image3]: ./plots/placeholder_small.png "Recovery Image"
[image4]: ./plots/placeholder_small.png "Recovery Image"
[image5]: ./plots/placeholder_small.png "Recovery Image"
[image6]: ./plots/placeholder_small.png "Normal Image"
[image7]: ./plots/placeholder_small.png "Flipped Image"

---
### Files Submitted 

[![Demo CountPages alpha](https://s3.eu-central-1.amazonaws.com/luca-public/udacity/behavioural-cloning/mountain_track.gif)](https://www.youtube.com/watch?v=ek1j272iAmc)


My project includes the following files:

* `models.py` containing the definition of the Keras models 
* `train.py` containing the training loop for the model
* `drive.py` for driving the car in autonomous mode (N.b. this file was modified to increase the speed of the car)
* `models/model.h5`, containing a trained convolution neural network which can be downloaded here 
* `data_pipe.py`, containing code for generators which stream the training images from disk and apply data augmentation
*  `fabfile.py`, fabric file which is used for automation (uploading code to AWS and download trained model)
* `README.md` summarizing the project

Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py models/model.h5
```

###Model Architecture and Training Strategy

#### Model architecture & Training Strategy

The project has developed in over few iterations. First, I established a solid data loading and processing pipeline to allow faster adjustment to the modeling part in the next steps.
For this reasons, some generators to stream training and validation data from disk were implemented in `data_pipe.py` in such a way that I could easily add more training and validation data.
The data is organized in the folders under `data`, each sub folder corresponds to different runs (explained later) and if a new folder is added this is automatically added 
by the generators to the training / validation loop.  Also, I wrote a fabric file to easily upload data and code to AWS, train the model and download the trained models definitions.

In order to establish the training and testing framework I initially used a very simple network with only one linear layer.
However, the final model architecture (`models.py` lines 56-93) is inspired to the architecutre published by Nvidia
self driving car team [!url](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) 
consisted of a convolution neural network with the following  layers and layer sizes.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param # 
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 70, 320, 3)        12
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 33, 158, 24)       1824
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 77, 36)        21636
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 37, 48)         43248
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 35, 64)         27712
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 2, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              4917900
_________________________________________________________________
dropout_1 (Dropout)          (None, 1164)              0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 5,171,331
Trainable params: 5,171,331
Non-trainable params: 0
_________________________________________________________________
```

Compared to the original Nvidia architecture, rectified linear units were used throughout the entire network except for
the initial and final layer.  Three other layers were added: `lambda_1` implements normalization by shifting the image values in `[-1,1]` 
and `cropping2d_1` removes the 65 and the bottom 25 pixels, while `conv2d_1` is a 3x3 convolutional layer with linear activation 
which is used to allow the model to learn automatically which transformation of intial RGB color space to be used 
(trick taken from ![url](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) )

The validation set helped determine if the model was over or under fitting. As explained in the data collection part 
the validation set consisted of two full laps around each track and was saparated from the training set. 
Qualitatively a lower `MSE` on the validation set seemed to correlate well with an improved driving behaviour of the car.
In order to reduce overfitting an aggressive dropout (droupout prob .30) was used between the last three final fully connected layers as well as data augmentation strategies (described later).
I used an adam optimizer so that manually  training the learning rate wasn't necessary and used early stopping with patience of 3 to decide the optimal number  of epochs (which happend to be around 10 most of the times).



###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

---

####3. Creation of the Training Set & Training Process

The training and validation data used can be downloaded here in zip format.

To capture good driving behavior, I first recorded two datasets each one consisting of one full lap on the track one using center lane driving. 
Here is an example image of center lane driving:

![alt text][image2]

I then recorded two other datasests the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 

These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


Since at the end my model was still drifting on a very difficult curve on the first track I added a few more frames which
would allow the model to learn the correct driving behaviour on that part of the track.

After the collection process, I had X number of data points. I then preprocessed this data by ...





