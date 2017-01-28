#Image_processing.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import regionprops
from skimage.measure import label
from skimage.exposure import histogram

import Constants as con

#get desired image
#(1)    ... RESIZE
#(2)    ... GRAYSCALE -> RESIZE
#(3)    ... THRESHOLD -> RESIZE
def load_and_get_images(full_path_to_image):
    path = full_path_to_image

    #get original image and resize it
    original_image = cv2.imread(path)
    original_image_resized = cv2.resize(original_image,
                                        (con.RESIZE_WIDTH,
                                         con.RESIZE_HEIGHT))

    #get original image and resize it
    #and grayscale it
    gray_image = cv2.imread(path,0)     #zero for grayscale
    gray_image_resized = cv2.resize(gray_image,
                                   (con.RESIZE_WIDTH,
                                    con.RESIZE_HEIGHT))

    img,t_image = cv2.threshold(gray_image_resized.copy(),127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,1),np.uint8)
    t_image_fix = cv2.morphologyEx(t_image,cv2.MORPH_OPEN,kernel)

    return original_image, original_image_resized, gray_image, gray_image_resized, t_image, t_image_fix

#Image_processing.py
#Import -- ip
