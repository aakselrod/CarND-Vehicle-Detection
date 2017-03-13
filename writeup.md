##Writeup Template
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-notcar.png
[image2]: ./output_images/hog_car.png
[image3]: ./output_images/test4_windows.jpg
[image4]: ./output_images/test6_windows.jpg
[image5]: ./output_images/test6_heat_labeled_windows.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first and second code cell of the IPython notebook.

I started in the first cell by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In the second cell, I adapted the functions developed in the teaching materials and exercises. The `get_hog_features()` function uses `skimage.hog()` to extract HOG features using specified parameters; the `bin_spatial()` function extracts spatial features; the `color_hist()` function extracts a color histogram, and the `extract_features()` function creates training data for training the classifier below based on settings in the third cell.

In the third cell, I then explored different color spaces and different HOG parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orient=9`, `pix_per_cell=(8, 8)` and `cell_per_block=(2, 2)` using the Cb channel:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. HSV seemed to give me the best accuracy score when testing the classifier, so at first I settled on that. However, when using the classifier below in a sliding window search, YCrCb gave me fewer false positives and was better able to see the white car in the example images (examples below). I tried to use a smaller number of pixels per cell, but that used too much RAM and slowed down the implementation; it did give me slightly better results but it wasn't worth the resource and performance hit.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I extracted features and created labels for training using the `extract_features()` function and scaled/normalized them using `sklearn.preprocessing.StandardScaler()` in the fourth code cell. In the fifth code cell, I trained a linear SVM `sklearn.svm.LinearSVC` with the features and labels. The accuracy of the classifier with the given parameters was almost 99%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the sixth code cell, I adapted functions from the class materials and exercises for performing a sliding window search. The `single_img_features()` function extracts spatial and color features from a single sub-image from a sliding window, with the default size being 64x64 as that's the size on which I trained the classifier. The `slide_window()` function slides a 64x64 window down and across an image with specified overlap and starting/stopping positions and returns the positions of those windows. The `draw_boxes()` function draws bounding boxes as specified in the arguments to signify the positions of found windows.

The `find_cars()` function accepts a start and stop Y parameter to eliminate false positives in the sky/trees and on the hood, as well as a list of window sizes and overlaps. It converts the JPG input into the same space as the PNG inputs used to train the classifier and then iterates over window sizes and overlaps. For each window size, it resizes the image to a size where the window size is equivalent to 64x64, the same image size used to train the classifier. Then it gets HOG information for the whole image, slides 64x64 windows with appropriate overlap over the required part of the image, and runs `single_img_features()` on each window as well as extracting the HOG features for that window. Then it runs the classifier trained above on each window, and adds the window to the list of found windows if the classifier says it's likely to be a car.

The seventh code cell is where I experimented with various window sizes, overlap sizes, and set the Y start and stop to prevent false positives in the sky/on the hood of the car. I played with the settings until I found one that detects the cars when they're close and has few false positives. A sample image found in the seventh cell is below:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (160x160 and 128x128) with an overlap of 0.75 using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here is another example image:

![alt text][image4]

I found that the biggest thing I could do to improve the performance of the classifier is choose the appropriate color space and window sizes. It took quite a bit of experimentation to find a combination that worked well on the actual test images and video, regardless of the performance of the classifier on a test set.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  I kept a bounded buffer of the positive detections from previous frames, dropping the earliest when the bounds of the buffer were reached.  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap, result of `scipy.ndimage.measurements.label()`, and bounding boxes overlaid on a test image:

![alt text][image5]

I experimented with the test video until I found a good setting for the number of frames to keep in the bounded buffer as well as the number of positive detections to use as a threshold.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that cars that are farther away require a smaller window size, which both increased false positives and greatly reduced the performance of my pipeline. In addition, other color spaces also increased detections of farther cars. The pipeline therefore does not attempt to detect cars that are far away, and fails at doing so. If I were to try to do it, I would use HOG features from more than one color space as well as smaller window sizes. I'd use a GPU-accelerated classifier in order to improve performance when using more features and smaller window sizes, such as perhaps a convnet, and I'd adjust the number of frames in the bounded buffer and the threshold for positive detections across frames appropriately.