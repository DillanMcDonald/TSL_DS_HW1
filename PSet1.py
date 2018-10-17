#Dillan McDonald
#Data Science Homework 1
#you're gonna hate me for this, but I'm gonna lump em all into one file
#rip

import numpy as np
from matplotlib import pyplot as plt
import cv2

#Digital Image Structure
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
image = cv2.imread('Image_1.png',-1)
cv2.imwrite('B-RGB.png',image[:, :, 0]) #writing just the blue channel
cv2.imwrite('R-RGB.png',image[:, :, 2]) #writing just the red channel
cv2.imwrite('G-RGB.png',image[:, :, 1]) #writing just the green channel
image_b = cv2.imread('B-RGB.png',-1)    #reading back in each one
image_r = cv2.imread('R-RGB.png',-1)
image_g = cv2.imread('G-RGB.png',-1)

plt.axis("off") #turn off axis
plt.imshow(image_b, cmap='gray')   #show the blue channel
plt.show()

plt.axis("off") #turn off axis
plt.imshow(image_r, cmap='gray')   #show the red channel
plt.show()

plt.axis("off") #turn off axis
plt.imshow(image_g, cmap='gray')   #show the green channel
plt.show()

#problem 5
img_x = 400 #x width of new image
img_y = 400 #y height of new image
image2 = np.zeros(((img_y,img_x,4)))    #make the new file matrix
i = 0
j = 0
k = 0
if img_y < image.shape[0] and img_x < image.shape[1]:
    while i<img_y:
        while j<img_y:
            while k < 4:
                image2[i][j][k] = image[i][j][k] #steal dem values
                k = k + 1
            k = 0
            j = j + 1
        j = 0
        i=i+1

plt.axis("off") #turn off axis
plt.imshow(image2/255)  #for some reason it needed to be divided by 255 to be normalized for RGB output
plt.show()

#Digital Image Characteristics
#Problem 1
image = cv2.imread('Image_1.png',0) #import the image as grayscale
plt.hist(image.ravel(),256,[0,256])
plt.show()

#Problem 2
image = cv2.imread('Image_1.png',0) #import the image as grayscale
image_2 = cv2.imread('Image_2.jpg',0) #import image_2 as grayscale
plt.hist(image.ravel(),256,[0,256],color = 'r')
plt.hist(image_2.ravel(),256,[0,256],color = 'b')
plt.show()

#Morphological Operations
#Problem 1
image = cv2.imread('Particles.jpg',0)#import the particles image

ret, image_thresh = cv2.threshold(image,110 ,255,cv2.THRESH_BINARY)  #threshold=110 best I guess would be just particles, and you could probably do that by optimizing the threshold value based on the number of circles it recognizes
plt.hist(np.histogram(image_thresh)) #not really sure why this histogram has values in the 40,000 area
plt.show()
plt.imshow(image_thresh) #actual thresholded image
plt.show()

#Problem 2
image = cv2.imread('Particles.jpg',0)#import the particles image
ret, image_thresh = cv2.threshold(image,110 ,255,cv2.THRESH_BINARY)  #threshold=110 best I guess would be just particles, and you could probably do that by optimizing the threshold value based on the number of circles it recognizes
plt.subplot(3,2,1)
plt.imshow(image_thresh) #actual thresholded image
plt.title("Binary Image")
plt.axis("off") #turn off axis

plt.subplot(3,2,2)
kernel = np.ones((4,4),np.uint8)#kernal for erosion - ones to remove white space, zero to remove dark space, size is amount of erosion
cv2.imwrite('img_er.png',cv2.erode(image_thresh,kernel,iterations = 1))
erosion = cv2.imread('img_er.png',0)
plt.imshow(erosion) #erosion image
plt.title("Erosion Image")
plt.axis("off") #turn off axis

plt.subplot(3,2,3)
kernel = np.ones((4,4),np.uint8)#kernal for dialte - ones to add white space, zero to add dark space, size is amount of dilation
cv2.imwrite('img_di.png',cv2.dilate(image_thresh,kernel,iterations = 1))
dilate = cv2.imread('img_di.png',0)
plt.imshow(dilate) #erosion image
plt.title("Dilate Image")
plt.axis("off") #turn off axis

plt.subplot(3,2,4)
kernel = np.ones((4,4),np.uint8)
cv2.imwrite('img_op.png',cv2.morphologyEx(image_thresh,cv2.MORPH_OPEN,kernel))
openim = cv2.imread('img_op.png',0)
plt.imshow(openim) #open image
plt.title("Open Image")
plt.axis("off") #turn off axis

plt.subplot(3,2,5)
kernel = np.ones((4,4),np.uint8)
cv2.imwrite('img_cl.png',cv2.morphologyEx(image_thresh,cv2.MORPH_CLOSE,kernel))
closeim = cv2.imread('img_cl.png',0)
plt.imshow(closeim) #close image
plt.title("Close Image")
plt.axis("off") #turn off axis

plt.show()
#Erosion is the best for determining the number of particles

#Problem 3
kernel = np.ones((4,4),np.uint8)
image = cv2.imread('Particles.jpg',0)#import the particles image
ret, image_thresh = cv2.threshold(image,110 ,255,cv2.THRESH_BINARY)  #threshold=110 best I guess would be just particles, and you could probably do that by optimizing the threshold value based on the number of circles it recognizes
image_thresh = cv2.erode(image_thresh,kernel,iterations = 1)#drop some dope erosion in this
ret, labels = cv2.connectedComponents(image_thresh)

print("Number of Particles: ",np.amax(labels)) #originally got 251, with Erosion got 510 which is a much more reasonable number as I guess-timated 550
