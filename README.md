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
To work with the camera images, the first step we must do is to correct the distortions. This is done by detecting the corners of chessboards. With the obtained corners and the 3d coordinates(z=0) the opencv function calibrateCamera returns lens distortion coefficients and camera matrix, which can, later on, be used to correct the distortions. All images are now corrected by calling the function undistort. The estimated coefficients and matrix are stored in a file, that not each time the calibration must be done.

Here is an example of a chessboard image with the distorted image and the corrected image:


distorted image            | distortion-corrected image
:-------------------------:|:-------------------------:
![](output_images/img1distorted.png)  |  ![](output_images/img1distorted.png)



## Pipeline test
---

### Undistorted Image

The first step of our pipeline is to undistort the images. The LaneFinder class are created by passing 
the camera matrix and the distortion coefficients as arguments to the class. They are later used in the method get_undistored_image,
where the actual image is corrected. We can clearly see the difference between the undistored and corrected image:

distorted image            | distortion-corrected image
:-------------------------:|:-------------------------:
![](test_images/straight_lines1.jpg)  |  ![](output_images/straight_lines1_undistored.jpg)

### Image Transformations

The next step is to extract the information's from the image to detect the lanes. This is done by applying different thresholds to color spaces. As for color spaces, the HLS and LAB are used. From the HLS color space, the saturation and lightness channels allow separating the lanes from the road. Since the lanes are yellow or white, they have a high saturation and lightness. To these channels are different gradient thresholds(magnitude, directions) applied and combined with saturation and lightness channel. 
Tests with various color spaces have shown, that the LAB color space provides good information about the lane position, especially in areas with shadow. These transformations are done in the method convert in the LaneFinder class. 

transformations        | warped transformations
:-------------------------:|:-------------------------:
![](output_images/straight_lines1_binary_mask.jpg)  |  ![](output_images/straight_lines1_warped_mask.jpg)

### Perspective Transform

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 592, 450      | 200, 0        | 
| 687, 450      | 1080, 0      |
| 1000, 660     | 1080, 720      |
| 280, 660      | 200, 720        |

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

 [link to my video result](./project_video_with_lanes.mp4)

          
## Discussion
---


