## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we are asked to use what we've learned about deep neural networks and convolutional neural networks to classify traffic signs. The goal is to train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we are asked to try out our model on images of German traffic signs that you find on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Project files
---

### Jupyter notebook: [Traffic_Sign_Classifier.ipynb](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)
Running the entire notebook without errors requires:
* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
The environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
* Downloading and unzipping the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) files in a folder named traffic-signs-data one level up from where the notebook is located.
* The file [signnames.csv](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/signnames.csv) containing the match between sign lable ID and full string description.

Running cells that use the trained model without going through the entire retraining require the files containing the model to be restored:
* [checkpoint](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/checkpoint)
* [traffic_signs.data-00000-of-00001](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/traffic_signs.data-00000-of-00001)
* [traffic_signs.index](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/traffic_signs.index)
* [traffic_signs.meta](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/traffic_signs.meta)

### Jupyter notebook exported as html: [Traffic_Sign_Classifier.html](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.html)

### Write up: [writeup.md](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/writeup.md)
This write up provides answers and explanations to the [project rubric points](https://review.udacity.com/#!/rubrics/481/view), including additional pictures:
* [data_set_histogram.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/data_set_histogram.png)
* [original.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/original.png)
* [grayscale.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/grayscale.png)
* [normalized.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/normalized.png)
* [softmax_probs.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/softmax_probs.png)
* [feature_map_layer1.png](https://github.com/schambon77/CarND-Traffic-Sign-Classifier/blob/master/feature_map_layer1.png)
