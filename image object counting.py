# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:20:36 2022

@author: Mahesh
"""
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#im = Image.open('E:/object couting/Image.jpeg')
image = cv2.imread('E:/object couting/Image.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray');

blur=cv2.GaussionBlur(gray, (11,11),0)
plt.imshow(blur, cmap='gray')

canny = cv2.canny(blur,30,150,3)
plt.imshow(canny, cmap='gray')

dilated=cv2.dilate(canny, (1,1), iterations=2)
plt.imshow(dilated, cmap='gray')

cnt, heirarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,  CV2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

plt.imshow(rgb)
print('coins in the image:',len(cnt))

'''

'''

import cv2
import numpy as np

# Read in the image in grayscale
img = cv2.imread('E:/object couting/Image.jpeg', cv2.IMREAD_GRAYSCALE)

# Determine which openCV version were using
if cv2.__version__.startswith('2.'):
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()

# Detect the blobs in the image
keypoints = detector.detect(img)
print(len(keypoints))

# Draw detected keypoints as red circles
imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display found keypoints
cv2.imshow("Keypoints", imgKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2

# The input image.
image = cv2.imread("E:/object couting/object image counting .jpg", 0)
#image = cv2.imread("images/Osteosarcoma_01_small.tif")
#Extract only blue channel as DAPI / nuclear (blue) staining is the best
#channel to perform cell count.
#image=image[:,:,0] 

#No need to pre-threshold as blob detector has build in threshold.
#We can supply a pre-thresholded image.

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 10000

# Filter by Color (black=0)
params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1

# Distance Between Blobs
params.minDistBetweenBlobs = 0

# Setup the detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

print("Number of blobs detected are : ", len(keypoints))


# Draw blobs
img_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_blobs)
cv2.imshow("Keypoints", img_with_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("particle_blobs.jpg", img_with_blobs)