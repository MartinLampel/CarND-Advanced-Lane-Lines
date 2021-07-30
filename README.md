# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



## The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


##Project Structure
---

* [`main.py`](main.py) : contains the code that test the pipeline and augments the video with lanes
* [`lanefinder.py`](lanefinder.py) : the LaneFinder class implementens the various steps for the lane detections
* [`line.py`](lanefinder.py) : the class Line is based on the suggested class from the course and stores various informations about the line
* [`utils.py`](utils.py) : in this module are various functions to extract different informatios from the channels or gradients by applying thresholds 

## Camera Calibration
---
distorted image            | distortion-corrected image
:-------------------------:|:-------------------------:
![](output_images/img1distorted.png)  |  ![](output_images/img1distorted.png)



## Pipeline test
---

### Undistorted Image

distorted image            | distortion-corrected image
:-------------------------:|:-------------------------:
![](test_images/straight_lines1.jpg)  |  ![](output_images/straight_lines1_undistored.jpg)

### Image Transformations

transformations        | warped transformations
:-------------------------:|:-------------------------:
![](output_images/straight_lines1_binary_mask.jpg)  |  ![](output_images/straight_lines1_warped_mask.jpg)

### Perspective Transform

image with polygon       | warped image with polygon
:-------------------------:|:-------------------------:
![](output_images/straight_lines1_poly.jpg)  |  ![](output_images/straight_lines1_warped.jpg)

### Lane Identification
![](output_images/straight_lines1_lanes.jpg

### Radius and Position

### Final Output

![](output_images/straight_lines1_final.jpg) 

## Project Video
---


https://github.com/MartinLampel/CarND-Advanced-Lane-Lines/blob/a8db922e72e7d4b8d9b4985f390502bf0a2fc3d7/project_video_with_lanes.mp4

          
## Discussion
---


