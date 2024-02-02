# Kamal Al Molk Robot

---


**Table of contents:**

1. [Introduction](#introduction)

2. [Hatching Folder](#hatching-folder)
    1. [functions](#functions)
    2. [hatching](#hatching)
    3. [Models](#models)
    4. [Outputs](#outputs)
    5. [Main Class](#main-class)

---

## Introduction

In this project, we want to build a robot ,that can scan an input image(normally face image) by opencv 
and other python libraries.
finally,it can draw that picture in form of black and white paint.

---

## Hatching Folder
this folder is for give an image as an input, and finally, output the compatible SVG and TXT file.

### functions
we have three classes in this section:
- [Find_levels Class](#find_levels)
- [Remove_backgroung Class](#remove_backgroung)
- [Sculpt Class](#sculpt)

#### <span style="color:yellow"> Find_levels </span> 
The provided Python code defines a function named "find_levels" that takes an image and a margin as input and returns levels. The function uses various libraries such as OpenCV (cv2), NumPy, Matplotlib, and SciPy to perform its task.
Here's a brief explanation of the function:

```
def find_levels(image, margin=40):
```

This function takes two arguments: image, which is expected to be a grayscale image, and margin, which has a default value of 40.
```
kde = stats.gaussian_kde(image.ravel()[image.ravel() <250])
```
A Gaussian kernel density estimator (KDE) is calculated for the image data. The ravel() function is used to flatten the image into a one-dimensional array. Only values less than 250 are considered for the KDE calculation.

```
xx =np . Linspace (0, 255, 100)
yy =kde (xx)
```

An array xx of 100 evenly spaced numbers between 0 and 255 is created. Then, the KDE function is evaluated at these points to generate the corresponding yy values.
```
valleys = find_peaks(-yy)[0]
valleys = xx[valleys].astype(int)
```
Peaks of the negative KDE curve are identified. These represent the valley points in the original curve. The indices of these peak points are used to extract the corresponding values from xx.
```
peaks = find_peaks(yy)[0]
peaks = xx[peaks].astype(int)
```
Similarly, peaks of the positive KDE curve are identified. These represent the peak points in the original curve.
Depending on the number of valleys found, the function calculates the levels differently. If no valleys are found, it assigns fixed levels. If one or two valleys are found, it performs certain calculations to determine the levels. If more than two valleys are found, it assigns the valley points as levels.
```
return lvls
```
Finally, the function returns the calculated levels.

#### <span style="color:yellow">Remove_backgroung</span> 

The provided Python code imports the alter_bg class from the "pixellib.tune.bg" module and defines a function named "remove_bg". This function uses PixelLib, a library for performing image and video segmentation, to change the background color of an image.
Here's a detailed explanation of the code:
```
import pixellib
from pixellib.tune_bg import alter_bg
```

These lines import the necessary modules. The pixellib module provides the functionality for image and video segmentation, while the "alter_bg" class from the  "pixellib.tune.bg" module is used for altering the background of images.
```
def remove_bg(file_name):
```

This line defines a function named remove_bg that takes one argument, file_name, which is expected to be the path to the image file whose background needs to be changed.
```
change_bg = alter_bg(model_type = "pb")
```
An instance of the  "alter_bg" class is created with the model type set to "pb". This indicates that a pre-trained model will be used for the background alteration.
```
change_bg.load_pascalvoc_model("models/xception_pascalvoc.pb")
```
The load_pascalvoc_model method is called on the change_bg object to load the pre-trained model. The model file is located in the "models" directory and is named "xception_pascalvoc.pb".
```
img = change_bg.color_bg(file_name, colors = (255,255,255), detect ="person")
```
The"color_bg" method is called on the"change_bg" object to change the background color of the image. The image file is specified by  file_name, the new color is white (RGB value (255,255,255)), and the object to be detected and kept in the image is a "person".
```
return img
```
Finally, the function returns the modified image.

#### <span style="color:yellow">sculpt</span> 

The provided Python code imports the OpenCV (cv2) and NumPy libraries and defines a function named sculpt. This function uses the "Haar Cascade classifier" in OpenCV to detect faces in an image and then modifies the image to highlight the detected faces.
Here's a detailed explanation of the code:
```
import cv2
import numpy as np
```
These lines import the necessary modules. The cv2 module provides the functionality for image processing, while the numpy module provides support for large multi-dimensional arrays and matrices, along with a large collection of mathematical functions to operate on these arrays.
```
def sculpt(image):
```
This line defines a function named sculpt that takes one argument, image, which is expected to be an image.
```
face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
``` 
An instance of the "CascadeClassifier" class is created with the "Haar Cascade" XML file specifying the frontal face. This is used to detect faces in the image.
```
img_c = image.copy()
```
A copy of the input image is made. This is done so that the original image is not modified during the face detection and modification process.
```
faces = face_cascade.detectMultiScale(image, 1.1, 4)
```
The "detectMultiscale" method is called on the "face_cascade" object to detect faces in the image. The image is scaled by a factor of 1.1 at each step, and the minimum size of the face is 4 pixels.
```
(x, y, w, h) = faces[0]
(x_p, y_p, x_pp, y_pp) = (max(x-w//2, 0), max(y-h//2, 0), min(x+w+w//2, image.shape[1]), min(y+h+h//2, image.shape[0]))
```
The bounding box coordinates of the first detected face are extracted and adjusted to fit within the image boundaries.
```
mask1 = np.zeros_like(image)
mask2 = np.zeros_like(image)
```
Two zero-filled masks are created with the same shape as the input image.
```
def l1(p0, p1):
   return p0 * (y_pp - (y+h))/x - (p1 - (y+h))
def l2(p0, p1):
   return (p0 - (x+w)) * (y+h - y_pp)/(x_pp - (x+w)) - (p1 - y_pp)
```
Two linear equations are defined based on the bounding box coordinates of the detected face. These equations are used to create the masks.
```
xv, yv = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
mask1 = np.ma.masked_less(l1(xv, yv), 0).mask
mask2 = np.ma.masked_less(l2(xv, yv), 0).mask
mask = mask1 | mask2
image[mask] = 255
```
"Meshgrids" are created for the image dimensions, and the masks are applied to the image. The masks are combined using the logical OR operation, and the pixels outside the mask are set to white (255).
```
return image[y_p:y_pp, x_p:x_pp]
```
Finally, the function returns the cropped image containing only the detected face(s).

---

### hatching
In this folder , we have our [Hatch class](#hatch), that it use to hatch the input and make the SVG file.
But before that, what is SVG file(or G-Code)?  
SVG, or Scalable Vector Graphics, is an XML-based vector image format used for defining two-dimensional graphics. It supports interactivity and animation, making it suitable for both web and print applications. 
SVG images are defined in a vector graphics format and stored in XML text files. This means they can be scaled in size without loss of quality .
SVG images can be searched, indexed, scripted, and compressed, which makes them versatile for various uses. They can be created and edited with text editors or vector graphics editors, and are rendered by most web browsers.

#### <span style="color:yellow">hatch</span>  

The provided Python code is a complex piece of software that uses several libraries such as matplotlib, numpy, shapely, skimage, svgwrite, and cv2 to manipulate images and generate SVG and G-Code files. It appears to be part of a larger program for creating hatch patterns on images and converting those patterns into machine instructions for a CNC router.
Here's a breakdown of the main functions:

	1. _build_circular_hatch: This function generates a circular hatch pattern. It takes parameters such as the hatch pitch, offset, width, height, and center of the hatch. It uses the numpy library to calculate the coordinates of the hatch lines and the shapely library to create the hatch pattern.

	2. _build_diagonal_hatch: This function generates a diagonal hatch pattern. Similar to the _build_circular_hatch function, it takes parameters such as the hatch pitch, offset, width, height, and angle of the hatch. It calculates the coordinates of the hatch lines and creates the hatch pattern.

	3. _plot_poly and _plot_geom: These helper functions are used to plot the hatch patterns. They take a geometric object and a color specification as inputs and plot the object using matplotlib.

	4. _build_mask: This function creates a mask for the hatch pattern. It takes a list of coordinates as input and uses the shapely library to create the mask.

	5. _save_to_svg: This function saves the hatch pattern as an SVG file. It takes the file path, width, height, and coordinates of the hatch lines as inputs and uses the svgwrite library to create the SVG file.

	6. _arrange: This function arranges the coordinates of the hatch lines based on an angle. It uses the numpy library to perform the arrangement.

	7. _get_coords: This function gets the coordinates of the hatch lines from a list of MultiLineString objects. It uses the shapely library to access the coordinates.

	8. _normalize: This function normalizes the coordinates of the hatch lines based on a configuration dictionary. It uses the numpy library to perform the normalization.

	9. _save_gcode: This function saves the hatch pattern as a G-Code file. It takes the file path, configuration dictionary, and coordinates of the hatch lines as inputs.

	10. _load_image: This function loads an image and applies optional transformations such as blurring, mirroring, and inversion. It uses the cv2 library to load and transform the image.

	11. _build_hatch: This function builds the hatch pattern based on an input image. It takes parameters such as the image, hatch pitch, levels, and whether the hatch is circular or not. It uses the _build_circular_hatch and _build_diagonal_hatch functions to build the hatch pattern.

	12. hatch: This is the main function that uses all the previously defined functions to create a hatch pattern based on an input image and save it as an SVG file and a G-Code file. It takes parameters such as the file path, output path, board configuration, hatch pitch, levels, blur radius, whether to arrange the hatch lines, whether to mirror the hatch lines horizontally, whether to invert the hatch lines, whether the hatch is circular or not, the center of the hatch, the angle of the hatch, whether to show a plot, whether to save as an SVG file, and whether to save as a G-Code file.


---


### Models

In this part , we put two web links for downloading some  requirements of this project, called [xception_pascalvoc.pb](#xception_pascalvoc) and [haarcascade_frontalface_default.xml](#haarcascade_frontalface_default). So , what do they do exactly?

#### <span style="color:yellow">xception_pascalvoc</span>  

The xception_pascalvoc.pb file is a pre-trained model used in the PixelLib library for image segmentation. The .pb extension indicates that it's a Protocol Buffers (protobuf) file, which is a method developed by Google for serializing structured data. In the context of machine learning and artificial intelligence, protobuf is often used to store trained models because it's a compact binary format that can be easily read and written by various languages .
Specifically, the xception_pascalvoc.pb file is a TensorFlow model that uses the Xception architecture and is trained on the Pascal VOC dataset. The Xception architecture is a deep convolutional neural network architecture introduced by Facebook. The Pascal VOC dataset is a popular dataset used for object detection, segmentation, and classification tasks.

In the PixelLib library, this model is used for changing the background of images. For example, in the provided code snippet, the load_pascalvoc_model method is used to load the xception_pascalvoc.pb model, and the color_bg method is used to change the background color of an image.
If you encounter an error message stating that the file signature could not be found when trying to open the xception_pascalvoc.pb file, it might mean that the file is corrupted or not properly downloaded. You may need to download the file again or check if the file is compatible with your current environment .

#### <span style="color:yellow">haarcascade_frontalface_default</span> 

The "haarcascade_frontalface_default.xml" file is a pre-trained Haar cascade classifier provided by OpenCV. Haar cascades are machine learning-based approaches where a cascade function is trained from a lot of positive and negative images. It is then used to detect the objects in other images. In this case, the haarcascade_frontalface_default.xml file is used for frontal face detection.
The Haar cascade classifier works by breaking down the detection process into stages, each consisting of a set of features. The key idea behind Haar cascade is that only a small number of pixels among the entire image is related to the object in concern. Therefore, it is essential to discard the irrelevant part of the image as quickly as possible.
To use this file in your code, you need to load it using the "cv2.CascadeClassifier()" function in OpenCV. 

---

### Outputs
Each time that we run our program, we can see its outputs here.And that output contains an SVG file, and a TXT file, that includes lines of G-Codes for hardware parts of project. 

---

### Main Class

In the main class,we have compressing and Convert method, that can process our input image, just like what we want:

    1. The script imports necessary modules such as os, json, cv2 (OpenCV, a library for image processing), and datetime. It also imports two custom modules, hatching and functions.

    2.the function compressing() takes file name and resize the image by PIL library,if our image is landscape, we shouldnt rotate it, but if not, we must rotate and then save it as 'compressed.jpg' file. 

    3. The function convert() takes three parameters: output of function compressing(), output_path, and params. This function seems to be responsible for converting an image file, applying transformations, and saving the result.

    4. Inside convert(), it opens a JSON configuration file and loads its contents into the config variable.

    5. It calls the remove_bg() function from the functions module to remove the background from the image. Then, it applies the sculpt() function to the image.

    6. It finds the levels in the image using the find_levels() function and stores them in the lvls variable.

    7. Depending on the image_scale parameter, it calculates the scaling factors for the x and y dimensions of the image. If the scale is set to 'auto', it scales the image proportionally based on the maximum dimension of the original image. Otherwise, it scales the image according to the given scale factor.

    8. It resizes the image using the calculated scaling factors and converts the color space from BGR to grayscale.

    9. It then calls the hatch() function from the hatching module to apply a hatch effect to the image. The hatch effect is configured according to the parameters passed to the function.

    10. In the main part of the script, it creates a directory named with the current timestamp under the outputs/ directory.

    11. It sets up the parameters for the image conversion, including the image scale and hatch angle.

    12. Finally, it calls the convert() function to process an image file named "th.jpg" and saves the output in the newly created directory.

In summary, this script processes an image by removing the background, applying a transformation, resizing it, converting it to grayscale, and applying a hatch effect. The processed image is saved in a new directory named with the current timestamp .



