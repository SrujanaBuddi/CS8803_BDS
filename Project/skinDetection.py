
'''
i.resize to 30x24
ii.fuzzy histogram equalization
iii.Prewitt filtering
iv. skin color segmentation
'''
# import the necessary packages
import cv2
import glob
import argparse
import numpy as np
from scipy import stats
from skimage import measure
import matplotlib.pyplot as plt
from neupy import algorithms
from collections import Counter
from scipy.misc import imresize
from pyzernikemoment import Zernikemoment

#returns uppper and lower hue limits
def adaptiveSkinColorDetection(image):
	lowerHue  = 0
	upperHue = 40
	h,s,v = cv2.split(image)
	#print image.shape
	histogram = np.histogram(h, normed = True)
	minValue, maxValue = getHistThresholds(histogram, 0.05, 0.05)
	return lowerHue, upperHue

lowerHue = 0
upperHue = 40
lower = np.array([lowerHue, 20, 50], dtype = "uint8")
upper = np.array([upperHue, 255, 255], dtype = "uint8")

#read all the images

#reading the image
#frame = cv2.imread('ASLsigns/C/172189195.jpg')
frame = cv2.imread('ASL.jpg')

frame = cv2.GaussianBlur(frame, (5, 5), 2)
cv2.imwrite("lowPass55_2.png",frame)

#resizing the image
frame = cv2.resize(frame, (100,100))
cv2.imwrite("resize.png",frame)

#adaptive histogram equalization for contrast enhancement
frame_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
cv2.imwrite("yuvspace.png",frame_yuv)
frame_yuv[:,:,0] = cv2.equalizeHist(frame_yuv[:,:,0])
cv2.imwrite("yuvhisteq.png",frame_yuv)
frame = cv2.cvtColor(frame_yuv,cv2.COLOR_YUV2BGR)
cv2.imwrite("yuv2bgr.png",frame)
#prewitt filtering


# apply a series of erosions and dilations to the mask
# using an elliptical kernel
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imwrite("bgr2hsv.png",converted)
skinMask = cv2.inRange(converted, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
skinMask = cv2.erode(skinMask, kernel, iterations = 2)
skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
cv2.imwrite("skinmask5_0.png",skinMask)
skin = cv2.bitwise_and(frame, frame, mask = skinMask)
# show the skin in the image along with the mask
cv2.imshow("images", np.hstack([frame, skin]))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("skin5_0.png",skin)

#getting conncetd components
labeled = measure.label(skin,connectivity=2)
cv2.imshow("images", labeled)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("skinConnctedComps.png",labeled)
