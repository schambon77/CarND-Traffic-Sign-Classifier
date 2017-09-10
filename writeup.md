
## Traffic Sign Classifier

### Goals
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

This write up relates to the following [rubric points](https://review.udacity.com/#!/rubrics/481/view) and complement the [Jupyter notebook](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Loading

The notebook assumes the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) files are located in a folder named traffic-signs-data one level up from the notebook. The 3 files, train.p, valid.p, test.p are loaded. For each, a features and labels array are populated.

### Data Set Summary and Exploration

#### Summary
Using the pandas library, a few basic numbers are computed from the train, valid and test features and labels arrays.
* Length of each set, 34799, 4410, 12630 respectively
* Feature size: 32x32x3, as they are color images
* Number of classes: 43

#### Exploration
A set of 25 pictures selected randomly from the training set are displayed, in order to get a feel for the images at a first glance.

Another picture is randomly selected and displayed again, with the aim to visualize subsequent pre-processing on this instance.

Finally, a histogram is displayed on each set, as well as the 3 sets merged, to get a feel for the spread of images across classes. One thing to note is the high variability in representation between classes. For the entire set, this goes from around 300 to about 3000, roughly a factor 10.

![Data Set Histogram][image1]

### Design and Test a Model Architecture

#### Preprocessing

The previously randomly selected original picture is shown here:

![Original Image][image2]

As a first step, I convert the images to grayscale using the OpenCV library, in order to reduce the dimensions for each pixel from 3 to 1, hence reducing the dimensions of each feature. This allows a simpler reuse of the LeNet model architecture. The converted image can be seen below:

![Grayscale Image][image3]

Then, I normalize the images using the preprocessing normalization function from the Scikit-Learn library. This allows a better spread of feature data points across the entire range. The same image after normalization can be seen below:

![Normalized Image][image4]

#### Model Architecture

I have started from the LeNet model and added 1 convolutional and 1 fully connected layers.

My final model consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x10 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 12x12x10 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten 		| outputs 400 samples        									|
| Fully connected		|  outputs 200 samples       									|
| RELU					|												|
| Fully connected		|  outputs 120 samples       									|
| RELU					|												|
| Fully connected		|  outputs 84 samples       									|
| RELU					|												|
| Fully connected		|  outputs 43 samples       									| 

Output logits and corresponding labels are then used to the overall loss with a softmax cross entropy function.

#### Model Training

To train the moel, I used:
* the AdamOptimizer, already used for the LeNet model training
* a default learning rate of 0.001
* a default batch size of 128
* 150 epochs

The training was performed on an AWS instance of g2.x2large type allowing usage of GPU to speed up training. Overall training was performed in a few minutes.

#### Solution Approach

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.9% 
* test set accuracy of 92.3%

To converge towards the solution, I started with the LeNet model architecture. I chose to add layers in order to better capture the additional details of the traffic signs compared to raw hand-written LeNet digits. I also assumed a higher number of epochs would be a good help given we have more samples in this data set. 

The training set accuracy was not computed as part of my trial and error process, but I realize this would have given me a good indication of under/overfitting. I primarily focused on the validation set accuracy, computed at each of my model training.

The test set accurcay was however computed only once after I was pleased with the validation results.

### Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

![Web sample 0][image5]
![Web sample 1][image6]
![Web sample 2][image7]
![Web sample 3][image8]
![Web sample 4][image9]

These appear to fairly good quality pictures, with little issues. However 3 images show a complex background which might bring challenges to the model.

#### Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop       		| Stop    									| 
| Turn right ahead     			| Turn right ahead 										|
| No entry					| No entry											|
| Speed limit (30km/h)	      		| Speed limit (30km/h)					 				|
| Roundabout mandatory			| Roundabout mandatory      							|

The model was able to predict correctly all 5 signs, for 100% accuracy on this fairly small set, comparing well with the test accuracy of 92.3%.

#### Model Certainty - Softmax Probabilities

As expected from the test set accuracy and quality of the web samples taken, the prediction probabilities are very high for the correct label and very small for others. As an example, the top 5 probabilities are displayed in the bar chart below:

![Top 5 probabilities][image10]

The second highest probability is in the order of 3e-18 !!!

The sign with the highest second probability is the Roundabout mandatory sign, with a second highest probability of about 2e-6. So very little doubt nonetheless.

### Visualizing the Neural Network

AS an additional excercise,  I have displayed the feature map of the 3 convolutional layers of my model. The feature map of the first layer is displayed below.

![Feature map layer 1][image11]

I can see from this feature map that the background is well ignored, and key features of the image are detected: round contour of the sign, numbers 3 and 0 from the 30km/h speed limit.   

[//]: # (Image References)

[image1]: ./data_set_histogram.png "Data Set Histogram"
[image2]: ./original.png "Original Image"
[image3]: ./grayscale.png "Grayscale Image"
[image4]: ./normalized.png "Normalized Image"
[image5]: ./web_samples/im_cropped_resized0.jpg "Web sample 0"
[image6]: ./web_samples/im_cropped_resized1.jpg "Web sample 1"
[image7]: ./web_samples/im_cropped_resized2.jpg "Web sample 2"
[image8]: ./web_samples/im_cropped_resized3.jpg "Web sample 3"
[image9]: ./web_samples/im_cropped_resized4.jpg "Web sample 4"
[image10]: ./sofmax_probs.png "Top 5 probabilities"
[image11]: ./feature_map_layer1.PNG "Feature map layer 1"

