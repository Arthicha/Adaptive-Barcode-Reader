'''
example: barcode reader
'''

import numpy as np
import cv2
import os
from barcode_reader import *

FOLDER = 'imgs' # input folder name
IMAGES = [filename for filename in os.listdir(FOLDER)] # testing image list
	

for image in IMAGES:

	# perform adaptive detection and decoding
	print("PROCESSING:",image)
	img = cv2.imread(FOLDER+"/"+image)
	code, intermediate_imgs = adaptive_read(img,imgsize=(4000,3000),detectionparams=(13,10,100),binarizationparams=(10,30,101))
	print("BARCODE:",str(code)[2:-1])

	# visualize the result
	names = ['gradient','threasholded gradient',"morph","detection","crop","binarization"]
	for i in range(len(intermediate_imgs)-1):
		cv2.imwrite("results/intermediate/"+image[:-4]+"_"+str(i)+"_"+names[i]+".jpg",intermediate_imgs[i])
	cv2.imwrite("results/final/"+image[:-4]+"_result"+".jpg",intermediate_imgs[-1])
	
