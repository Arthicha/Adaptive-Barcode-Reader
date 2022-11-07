# Adaptive Barcode Reader

This git repository provides an adaptive barcode detection and decoding function for robust barcode detection and decoding. The project is a part of expert in team inovation project (E22).

## Method
The flowchart of the provided example program is presented in the figure below. The "adaptive barcode detection and decoding" block denotes the "adaptive_read()" function imported from "barcode_reader.py".

![](document/method.png "Method")


# Installation 

To install the required python modules (standard modules, such as numpy and opencv, not included), run ` pip install -r requirement.txt `.


# Runing The Example
## Running an example program
To run an example code, please run ` python example.py `. The example program will perform adaptive barcode detection and decoding on all images locating in the "imgs" folder.
## Running the markdown documentation
To view the readme, please run ``` grip -b readme.md ```.

# Adaptive Barcode Detection and Decoding
### python code:
```
from barcode_reader import *
img = <input image>
code, intermediate_imgs = adaptive_read(img, imgsize=(<width>,<height>), detectionparams=(<k>,<dk>,<kmax>), binarizationparams=(<th>,<dth>,<thmax>))
```
### adaptive_read()
**File:** barcode_reader.py

**Description:** perform adaptive barcode detection and decoding

**Input:**
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

**Output:**
1. code (string): barcode data
2. intermidiate_imgs (list of images): intermediate images

# Result
**test1.jpg**
![](imgs/test1.jpg)
![](results/final/test1_result.jpg)
**test2.jpg**
![](imgs/test2.jpg)
![](results/final/test2_result.jpg)
**test3.jpg**
![](imgs/test3.jpg)
![](results/final/test3_result.jpg)
**test4.jpg**
![](imgs/test4.jpg)
![](results/final/test4_result.jpg)
**test5.jpg**
![](imgs/test5.jpg)
![](results/final/test5_result.jpg)
**test6.jpg**
![](imgs/test6.jpg)
![](results/final/test6_result.jpg)
**test7.jpg**
![](imgs/test7.jpg)
![](results/final/test7_result.jpg)
**test8.jpg**
![](imgs/test8.jpg)
![](results/final/test8_result.jpg)
**test9.jpg**
![](imgs/test9.jpg)
![](results/final/test9_result.jpg)

# Reference
This code is modified from [this git repository](https://github.com/pyxploiter/Barcode-Detection-and-Decoding).
