#Recognition.py

#Recognition.py should do the following (ALPHA BUILD):
#   (1) -   Load image with license plate (LPI)
#   (2) -   Process LPI to grayscale, threshold, and resize the images
#   (3) -   Extract regions of LPI
#   (4) -   Show regions of LPI

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import regionprops
from skimage.measure import label
from skimage.exposure import histogram

import Constants as con
import Image_processing as ip
import Image_presentation as impress
import KNN as kn

#load LPI and process it:
#   (1) ..  Resize the original image and keep original size (LPI_original, LPI_original_resized)
#   (2) ..  Resize the grayscale original and keep original size (LPI_gray, LPI_gray_resized)
#   (3) ..  Resize the threshold of grayscale image and keep original size (LPI_thresh, LPI_thresh_resized)
#   (4) ..  Fix the thresholded images with a binary ones opening kernel
#           (LPI_t_fix, LPI_tr_fix)
#   (5) ..  Return original, original resized, grayscale, grayscale resized, thresh fix, thresh fix resized
def load_and_process_LPI(image_name):
    LPI_original = cv2.imread(image_name)
    LPI_original_resized = cv2.resize(LPI_original,
                                      (con.REC_RESIZE_WIDTH,
                                       con.REC_RESIZE_HEIGHT))

    LPI_gray = cv2.imread(image_name,0)
    LPI_gray_resized = cv2.resize(LPI_gray,
                                  (con.REC_RESIZE_WIDTH,
                                   con.REC_RESIZE_HEIGHT))

    ret1, LPI_thresh = cv2.threshold( LPI_gray.copy(),
                                      127,
                                      255,
                                      cv2.THRESH_BINARY_INV)
    ret2, LPI_thresh_resized = cv2.threshold( LPI_gray_resized.copy(),
                                     127,
                                     255,
                                     cv2.THRESH_BINARY_INV)

    kernel = np.ones((1,1),
                     np.uint8)

    LPI_t_fix = cv2.morphologyEx( LPI_thresh,
                                  cv2.MORPH_OPEN,
                                  kernel)

    LPI_tr_fix = cv2.morphologyEx( LPI_thresh_resized,
                                   cv2.MORPH_OPEN,
                                   kernel)

    return LPI_original, LPI_original_resized, LPI_gray, LPI_gray_resized, LPI_t_fix, LPI_tr_fix
    

def extract_regions_LPI(LPI_thresh):
    LPI_labeled, LPI_regions = ip.get_labeled_regions(LPI_thresh)
    return LPI_labeled, LPI_regions

def show_regions_LPI(LPI_original, LPI_regions):
    impress.draw_regions_on_image("Regions",LPI_original,LPI_regions)

def prep_for_recognition(LPI_regions,LPI_thresh,LPI_original,TEST_FLAG):
    cropped_region_images = []
    flat_images = []
    
    #cropped out all regions
    cropped_region_images = ip.get_cropped_images(LPI_regions,LPI_thresh)

    #just to show all cropped images
    if TEST_FLAG == 1:
        titles = []
        for i in range(len(cropped_region_images)):
            titles.append(str(i))
        impress.show_multiple_images(titles,cropped_region_images)

    return cropped_region_images

#Recognition.py
