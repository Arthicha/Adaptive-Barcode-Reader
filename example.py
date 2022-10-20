'''
example: barcode reader
'''

import numpy as np
import cv2
from barcode_reader import *

FOLDER = 'imgs'
IMAGES = ['test1','test2','test3','test4']

for image in IMAGES:
	print(FOLDER+"/"+image+".jpg")
	img = cv2.imread(FOLDER+"/"+image+".jpg")
	barcode, coord, imgs = bardetect(img)
	code, thresholded_code = bardecode(barcode)
	imgs.append(thresholded_code)
	names = ['gradient','threasholded gradient',"morph","detection","crop","binarization"]
	for i in range(len(imgs)):
		cv2.imwrite("results/"+image+"_"+str(i)+"_"+names[i]+".jpg",imgs[i])
	img = cv2.putText(img, str(code), coord, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imwrite("results/"+image+"_"+str(i)+"_"+"code"+".jpg",img)
	
