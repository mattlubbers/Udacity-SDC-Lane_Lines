# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---
While driving, humans perceive and understand that the proper paths to follow are indicated by traffic annotations in the form of lines on the ground. As drivers, we have been trained to monitor and adhere to these this guiding lines, however as Engineers, we must also construct a perception pipeline for a robot to detect and maintain these in a similar fashion.

This project aims to detect lane lines in pre-selected images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

Implementation
---

#### Perception Pipeline

The perception pipeline was created by using a few existing OpenCV functions, as well as constructing new functions that are necessary to create the weighted lane lines for the set of reference images. This was first performed by stepping through each layer of the perception pipeline with a single test image and tuning parameters to get the desired output result.

The perception pipeline consists of:
- Grayscale filtering
- Gaussian Blur
- Canny Edge
- Vertice mask
- Hough Lines
- Input image with hough line overlay


**Grayscale Filtering and Gaussian Blur**
Apply a Grayscale transform, and a Gaussian Noise Kernel using the OpenCV library:
```
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
Image of Grayscale and Gaussian Blur:
![grayGaussian](/assets/grayGaussian.png)

**Canny Edges**

Define the canny transform with the the OpenCV library, and call the function with the appropriate lower and upper thresholds for edge detection:
```
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

ip_canny = canny(ip_gaussian, 60, 120)
```
To determine the proper canny edge parameters, the image was processed with multiple parameters, and plotted in an output grid for easy comparison [(10,30) , (30,90) , (50,150) , (60,120)]:
![canny](/assets/canny.png)

**Verticies Mask**

The lane lines will consistently appear in a specific field of interest, so to reduce computational resources, let's filter out all other regions:
```
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#Set mask verticies to limit range of interest within image
vertices = np.array([[(90,image.shape[0]),(450, 330), (510, 330), (image.shape[1],image.shape[0])]], dtype=np.int32)

#Call the function
ip_mask = region_of_interest(ip_canny, vertices)
```
![mask](/assets/mask.png)

**Hough Lines**

Define functions to create lines using y = mx + b:
```
def line_fit(img, x, y, color=[255, 0, 0], thickness=20):
    fit = np.polyfit(x,y,1)
    m, b = fit
    
    y_1 = img.shape[0]
    y_2 = int(y_1 / 2) + 50
    # y = mx + b ----> x = (y - b) / m
    x_1 = int((y_1 - b) / m)
    x_2 = int((y_2 - b) / m)
    cv2.line(img, (x_1, y_1), (x_2, y_2), color, thickness)
    
def draw_lines(img, lines):
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = (y2 - y1) / (x2 - x1)
            if m < 0:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
            else:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
    
    line_fit(img, left_x, left_y)
    line_fit(img, right_x, right_y)
```
Now define the hough line function with the hough parameters for thresholds. Also, draw the hough lines ontop of the canny processed image:
```
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img
```
![hough](/assets/hough.png)

**Hough Lines on Input Image Overlay**
Overlay the processed image ontop of the input image with a specified alpha and beta parameter:
```
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)
```
![final](/assets/final.png)

After these parameters were tuned, the entire dataset of test images was processed through the perception pipeline to create output images. Furthermore, this perception pipeline was then used to construct weighted lane lines for a continuous video, to indicate the robustness and accuracy of this processing method.


The first step is to apply a grayscale filter and a Gaussian Blur to the original input image:

Reflection
---
**Shortcomings**
The hough lines overlayed with the input images matched fairly well, however they were not a perfect match for thickness, nor the angle alignment. Furthermore, upon playing the video of the continuous image processing, it is clear that the current parameters selected were approximately 80% accurate, and resulted in large angle discrepencies that were unreasonable for lane lines. 

**Possible Improvements**
- Increase the accuracy within the continuous video perception processing. This will likely require more parameter tuning, and potentially more filters
- A wider range of test images (light conditions, faded markings, or arrows) would provide additional robustness
- It may be beneficial to set a limit for the min/max angle of the line. For example, we would not ever expect a line with a magnitude of 180 degrees (horizontal), but 75 degrees seems feasible
- Similarly, we could also specify the max delta between the current lane angle and the previous. It would be unrealistic to believe that the lane line angle "jumps" from 75 degrees to 180 degrees during the transition of a single frame
