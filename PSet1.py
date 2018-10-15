#Dillan McDonald
#Data Science Homework 1

import numpy
from matplotlib import pyplot as plt
from matplotlib import image as img
import cv2

#problem 1
#import the image as RGB
image = cv2.imread('Image_1.png',-1)
plt.axis("off") #turn off axis
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))   #convert the openCV BGR to RGB for matplotlib to properly export
plt.show()

#problem 2
#import the image as greyscale
image = cv2.imread('Image_1.png',0)
plt.axis("off") #turn off axis
plt.imshow(image,cmap='gray')   #make sure that it plots in greyscale
plt.show()

#problem 3
#import the image as RGB
image = cv2.imread('Image_1.png',-1)
plt.axis("off") #turn off axis
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), cmap='gray')   #convert the openCV BGR to gra
plt.show()

#problem 4



