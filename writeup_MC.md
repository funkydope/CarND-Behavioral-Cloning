# **Behavioral Cloning** 

## Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia.png "Nvidia Architecture"
[image2]: ./writeup_images/off_right.jpg "Driving off Right Guide Line"
[image3]: ./writeup_images/off_left.jpg "Driving off Left Guide line"
[image4]: ./writeup_images/center.jpg "Center Camera Image"
[image5]: ./writeup_images/left.jpg "Left Camera Image"
[image6]: ./writeup_images/right.jpg "Right Camera Image"
[image7]: ./writeup_images/center_flipped.png "Center Camera Flipped Image"
[image8]: ./writeup_images/left_flipped.png "Left Camera Flipped Image"
[image9]: ./writeup_images/right_flipped.png "Right Camera Flipped Image"
[image10]: ./writeup_images/center_crppped.png "Center Camera Cropped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_nvdia_gen_s01_d05_e5.h5 containing a trained convolution neural network 
* writeup_mc.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_nvdia_gen_s01_d05_e5.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (see model.py line 100). The model architecture is summarized here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/.

![alt text][image1]

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer with value 0.5 after the last convolutional layer to prevent overfitting. 

I used the sklearn API to split the complete data set into training and validation sub data sets.

#### 3. Model parameter tuning

The model used an Adam optimizer, which sets the learning rate automatically.

#### 4. Appropriate training data

I used 10 laps of simulator driving to train the model. 5 in the default counter-clockwise direction and in the reverser clockwise direction.

Dee the following section for more detail.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My model architecture strategy followed the Udacity lessons. I started with a 1 layer model confirm my code worked properly. I then followed the Nvidia Architecture. I thought this would work well since Nvidia's model operates with similar inputs.

The model converges quickly, usually in 1-5 epochs. At this point, the training loss continues to decrease while the validation loss stabilizes or increases. This shows that the model is starting to overfit the training data. I typically started the run the model training with 5 epochs to evaluate overfitting as shown below.

Epoch 1/5 <br> 
51258/51240 [==============================] - 295s - loss: 0.0089 - val_loss: 0.0067 <br>
Epoch 2/5 <br>
51265/51240 [==============================] - 265s - loss: 0.0078 - val_loss: 0.0064 <br>
Epoch 3/5 <br>
51261/51240 [==============================] - 265s - loss: 0.0075 - val_loss: 0.0071 <br>
Epoch 4/5 <br>
51244/51240 [==============================] - 265s - loss: 0.0076 - val_loss: 0.0063 <br>
Epoch 5/5 <br>
51267/51240 [==============================] - 265s - loss: 0.0071 - val_loss: 0.0064 <br>

In this case, I re-ran the training for 2 epochs since the validation loss does not decrease consistently after this point.

Finally, I would test the model on the driving simulator. My early models had trouble on the first sharper curve in front of the bridge. It would drive off the right side into the water. I also found that a steering correction of 0.2 for left and right images was too high. This caused the vehicle steering to be too busy. Reducing it to 0.1 yielded better driving.

I used Keras cropping to help the model focus only on the road. The upper and lower parts of the image are not useful to steering.

#### 2. Final Model Architecture <br>

| Layer             | Spatial Size| Depth|    Description	        					| 
|:---------------:  | :------:  | :---:|  :---------------------------------------------:| 
| Input         	| 160x320   | 3    |RGB Image   | 
| Cropping2D 		| 160x225   |	3  | 70 top, 25 bottom 	|
| Lambda 	        | 160x225   |  3   | normalize input to 0 mean, 0.5 range 	|
| 1 Convolution2D   |5x5 filter | 24  | relu activation, same padding   |
|   MaxPooling2D    |2x2        |     |   |
| 2 Convolution2D   |5x5 filter | 36  | relu activation, same padding   |
|   MaxPooling2D    |2x2        |     |   |
| 3 Convolution2D   |5x5 filter | 48  | relu activation, same padding   |
|   MaxPooling2D    |2x2        |     |   |
| 4 Convolution2D   |3x3 filter | 64  | relu activation, same padding   |
|   MaxPooling2D    |2x2        |     |   |
| 5 Convolution2D   |3x3 filter | 64  | relu activation, same padding   |
|   Flatten    |1        |     | Flatten last convolutional layer   |
|   Dropout    |1        |     | value 0.5  |
| 6 Dense pooling| 1 |1 | Single Fully Connected Layer, outputs steering angle|

My model differs from Nvidia's in the following ways:
-My model uses same padding rather than valid. This keeps the edge pixels intact
-My model only has one fully-connected layer after flattening
--Research has shown that additional fully-connected layers do not improve model performance.

Although my model stays on the road for an complete lap, it has trouble in 2 sections. The model drives onto the yellow guide lanes but does not drive off the road.

![End of Turn 1 - decreasing radius left turn, before the bridge][image2]

![End of Turn 3 - only right turn][image3]

I could improve the model's driving behavior by ensuring that only center driving data for these sections are in the training data set.

#### 3. Creation of the Training Set & Training Process

I tried my best to stay in the center of the lane all the training data. I found that using a keyboard or mouse did not allow to me to control the simulator well. I instead used a PlayStation controller to control the car. I could now stay in the center of the lane better and steering more smoothly.

I used left and right camera images along with the center image to improve training. The left and right images help the model train for off center driving scenarios. I corrected the steering angle for left and right images by 0.1.

Left, Center, and Right Cameras

![Left Camera Image][image5] ![Center Camera Image][image4] ![Right Camera Image][image6]

Since the the data for this model is very small, I flipped images to help the model generalize. The lap is counterclockwise with mainly left turns. This could bias the model to turning left. I now had 6 pieces of training data for each driving sample (center, left, right, flipped center, flipped, left, flipped right)

Flipped Images

![Flipped Left Camera Image][image8] ![Flipped Center Camera Image][image7] ![Flipped Right Camera Image][image9]

The camera images have a lot of information that isn't relevant to keeping the car on the road. The upper part of the image shows only the sky and background scenery. The bottom portion shows the car. Cropping these parts of the image helped the model train better.

Cropped Center Image
![Cropped Center Camera Image][image10]

After the collection process, I had 51240 number of data points.

My AWS EC2 instance had trouble handling the memory requirements for training the model. I created a generator function in my model.py code to process and feed images for each individual batch. It no longer needed to store all images at once.
