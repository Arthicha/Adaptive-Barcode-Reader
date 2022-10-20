'''
barcode reader functions
'''

# import the necessary packages
import numpy as np
import argparse
import cv2
from copy import deepcopy
from pyzbar.pyzbar import decode


# --------------------  warp transform --------------------
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# --------------------  barcode decoder --------------------
def bardecode(image):
	detectedBarcodes = []
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for th in range(254,10,-10):
		(_, thresh) = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
		#cv2.imshow("th"+str(th), thresh)
		detectedBarcodes = decode(thresh)

		if detectedBarcodes != []:
			break
	for barcode in detectedBarcodes:
		(x, y, w, h) = barcode.rect
		cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 5)
	  
		return barcode.data, thresh

# --------------------  barcode detector --------------------
def bardetect(image):
	output_images = []
	#resize image
	image = cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)

	#convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	#calculate x & y gradient
	gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
	gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

	# subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)
	output_images.append(gradient)
	# blur the image
	blurred = cv2.blur(gradient, (3, 3))

	# threshold the image
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
	output_images.append(thresh)

	# construct a closing kernel and apply it to the thresholded image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))#(21, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	# perform a series of erosions and dilations
	closed = cv2.erode(closed, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)
	output_images.append(closed)

	# find the contours in the thresholded image, then sort the contours
	# by their area, keeping only the largest one
	cnts,hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

	# compute the rotated bounding box of the largest contour
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))
	coord = np.min(box,axis=0)
	# draw a bounding box arounded the detected barcode and display the
	# image
	imagedetected = deepcopy(image)
	cv2.drawContours(imagedetected, [box], -1, (0, 255, 0), 3)
	output_images.append(imagedetected)

	barcode = four_point_transform(image,box)
	barcode = cv2.resize(barcode,None, fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
	output_images.append(barcode)

	return barcode, coord, output_images
