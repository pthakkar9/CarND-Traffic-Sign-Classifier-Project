# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[readme1]: ./examples/readme1.png "Visualization"
[readme2]: ./examples/readme2.png "Grayscaling"
[readme3]: ./examples/readme3.png "Normalized"
[readme4]: ./german-signs-data/z101.gif "Germen Traffic Sign"
[readme5]: ./german-signs-data/z123.gif "Germen Traffic Sign"
[readme6]: ./german-signs-data/z136_10.gif "Germen Traffic Sign"
[readme7]: ./german-signs-data/z205.gif "Germen Traffic Sign"
[readme8]: ./german-signs-data/z206.gif "Germen Traffic Sign"
[readme9]: ./german-signs-data/z274.gif "Germen Traffic Sign"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

[Project code is here](https://github.com/pthakkar9/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[HTML](https://github.com/pthakkar9/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html) and [PDF](https://github.com/pthakkar9/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.pdf) versions of jupyter notebook are available.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The training set, validation set and the test set has 43 different classes representing the german traffic signs. Here are the 10 random images from the data set.

![alt text][readme1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to convert the images to grayscale because of the same reasons we converted lane line project images to gray scale. Converting images to grayscale negates the visual differentiation of the images. Adding normalization makes model efficient as there are less parameters for the model to learn.

Here is an example of a traffic sign image with slowly offseting to grayscale.

![alt text][readme2]

Here is an example of an grayscale image and an augmented image:

![alt text][readme3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I reused code from lab excercises.

There are two convolution layers and three fully connected layers, with the last being the logits. In between layers 2 and 3, the image gets flattened.

Layer 1 to 2 takes an input of 32x32x1, uses VALID padding, and outputs to a nuerons with dimensions 28x28x6. The convolution filter had shape 5x5x1x6 and the stride was 1 and max pooling was performed.

Layer 2 to 3 takes the input of 28x28x6, uses VALID padding, and outputs to nuerons with dimensions 10x10x16. The covolution filter had shape 1x2x2x1 and the stride was 1 and max pooling was performed.

Layer 3 to 4 takes the input of 10x10x16, uses VALID padding, and outputs to nuerons with dimensions 5x5x16. The covolution filter had shape 1x2x2x1 and the stride was 1 and max pooling was performed.

Next, the image was flattnened and the layers were fully connected. Layer 3 to 4 The input was 5x5x16 which was output to 400 nuerons. Layer 4 to 5 takes 400 nuerons and connects to 120 nuerons. Layer 5 to output, the last layer connects the 84 nuerons to the 43 output classes.

The activation function used in each nueron was the Rectified Linear Unit.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained model on laptop - HP ENVY 17. It has 16 GB ram and i7 processor.

Here are the hyperparameters I landed on - 

`
EPOCHS = 50
BATCH_SIZE = 156
rate = 0.00097
mu = 0
sigma = 0.1
`

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.7% (**Changed after review**)
* validation set accuracy of 93.6% (**Changed after review**)
* test set accuracy of 92%

This solution is based on the LeNet model architecture. This model is to be proven effective in classifying the images. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][readme4] ![alt text][readme5] ![alt text][readme6] 
![alt text][readme7] ![alt text][readme8] ![alt text][readme9]

All six signs were taken from german tour guide site. They are not the pictures but real signs in .gif format. All six of them were part of the training set. 

As these signs are not picture and part of training format, all of them should be easy to classify for the model. If I have to guess, last image might run into some problems because 6 might reas as 5 and it could be classifed as 50 km/h image. (**Added after review**)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution  		| General Caution								| 
| Road work    			| Road work										|
| Children crossing		| Children crossing								|
| Yield		      		| Yield							 				|
| Speed limit (60km/h)	| Speed limit (50km/h) 							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

`
predicted signs:
[(18, b'General caution') (26, b'Traffic signals') (12, b'Priority road')
 (27, b'Pedestrians') (40, b'Roundabout mandatory')]
with corresponding probabilities: 
[  1.00000000e+00   6.11154349e-10   1.02727260e-18   4.94604789e-19
   9.91444077e-20]

predicted signs:
[(25, b'Road work') (20, b'Dangerous curve to the right')
 (38, b'Keep right') (36, b'Go straight or right') (23, b'Slippery road')]
with corresponding probabilities: 
[  1.00000000e+00   1.10255146e-20   6.15212405e-26   9.59336039e-28
   7.91629206e-32]

predicted signs:
[(28, b'Children crossing') (20, b'Dangerous curve to the right')
 (41, b'End of no passing') (32, b'End of all speed and passing limits')
 ( 0, b'Speed limit (20km/h)')]
with corresponding probabilities: 
[  9.85252142e-01   1.47478916e-02   3.76155712e-16   5.44196897e-17
   3.43958606e-17]

predicted signs:
[(13, b'Yield') (35, b'Ahead only') (36, b'Go straight or right')
 (12, b'Priority road') ( 0, b'Speed limit (20km/h)')]
with corresponding probabilities: 
[  1.00000000e+00   5.98428144e-20   3.30372981e-32   1.56983333e-38
   0.00000000e+00]

predicted signs:
[(14, b'Stop') ( 4, b'Speed limit (70km/h)') ( 2, b'Speed limit (50km/h)')
 ( 1, b'Speed limit (30km/h)') (38, b'Keep right')]
with corresponding probabilities: 
[  1.00000000e+00   2.57240301e-13   2.16597397e-13   4.07860933e-26
   3.21921686e-26]

predicted signs:
[( 2, b'Speed limit (50km/h)') ( 1, b'Speed limit (30km/h)')
 (31, b'Wild animals crossing') ( 3, b'Speed limit (60km/h)')
 ( 5, b'Speed limit (80km/h)')]
with corresponding probabilities: 
[  9.99995351e-01   4.68261851e-06   1.99782804e-13   3.66856954e-19
   2.06883229e-24]

`


================================================================



## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

