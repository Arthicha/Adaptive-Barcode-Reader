'''
barcode reader functions
'''

# import the necessary packages
import numpy as np
import argparse, sys
import cv2
from copy import deepcopy
from pyzbar.pyzbar import decode

# --------------------  detect and decode --------------------
def adaptive_read(img,imgsize=(4000,3000),detectionparams=(13,10,200),binarizationparams=(10,20,101)):
	'''
	Perform adaptive barcode detection and decoding
	input:
		1. img (opencv array): an image of an arbitrary size
		2. imgsize (tuple (w,h)): the desired size (width,height)
		3. detectionparams (tuple (k,dk,kmax)): the adaptive detection parameters
			- k (int): initial morphological transformation kirnel size
			- dk (int): morphological transformation kirnel step size
			- kmax (int): maximum morphological transformation kirnel step size
		4. binarizationparams (tuble (th,dth,thmax)): adaptive binarization parameters
			- th (int): initial threshold
			- dth (int): threshold step
			- thmax (int): maximum threshold
	output: 
		1. code (string): barcode data
		2. intermidiate_imgs (list of images): intermediate images
	'''
	img = cv2.resize(img,imgsize, interpolation = cv2.INTER_CUBIC)

	found = False
	code = None
	imgs = []
	for kernelsize in range(detectionparams[0],detectionparams[2],detectionparams[1]): # adaptive loop
		try : 			
			barcode, coord, imgs, rect = bardetect(img,kernelsize=kernelsize) # barcode detection
			code, thresholded_code = bardecode(barcode,thinit=binarizationparams[0],thmax=binarizationparams[2],thstep=binarizationparams[1]) # barcode decoding (rise an error if decoding fails)
			found = True
			break
		except:
			pass

	if found:
		imgs.append(thresholded_code)
		cv2.drawContours(img, [rect], -1, (0, 0, 255), 5)
		img = cv2.putText(img, str(code)[2:-1], (0.9*np.array(coord)).astype(int), cv2.FONT_HERSHEY_SIMPLEX,int(5), (255,255, 255), int(50), cv2.LINE_AA)
		img = cv2.putText(img, str(code)[2:-1], (0.9*np.array(coord)).astype(int), cv2.FONT_HERSHEY_SIMPLEX,int(5), (0, 0, 255), int(10), cv2.LINE_AA)
		imgs.append(img)
	intermidiate_imgs = imgs
	return code, intermidiate_imgs

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
def bardecode(image,thinit=10,thmax=101,thstep=20):
	detectedBarcodes = []
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	for th in range(thinit,thmax,thstep):
		thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,th+1,2)
		detectedBarcodes = decode(thresh)
		if detectedBarcodes != []:
			break

	for barcode in detectedBarcodes:
		return barcode.data, thresh

# --------------------  barcode detector --------------------
def bardetect(org_image,kernelsize=13):
	output_images = []
	#resize image
	image = deepcopy(org_image)

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
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelsize,5))#(21, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	# perform a series of erosions and dilations
	kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernelsize/2),5))
	closed = cv2.erode(closed, kernel2)
	closed = cv2.dilate(closed, kernel2)
	closed = cv2.erode(closed, kernel2)
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
	barcode = four_point_transform(org_image,box)
	barcode = cv2.resize(barcode,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	output_images.append(barcode)
	return barcode, coord, output_images, box
