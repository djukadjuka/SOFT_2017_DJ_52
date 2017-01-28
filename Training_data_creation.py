#Training_data_creation.py

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

TESTING = 1
IMAGES = [0]

def startup():

    images = []

    if TESTING == 1:                #if in testing mode
        for i in IMAGES:                #get only specific images
            images.append(con.TRAINING_DATA_IMAGE_NAMES[i])
    else:
        images = con.TRAINING_DATA_IMAGE_NAMES  #get all images

    for char in images:
        image_path = con.TRAINING_DATA_FOLDER_NAME+char+con.TRAINING_DATA_IMAGE_SUFFIX

        #getting the processed images
        original_image,gray_image,thresholded_image = ip.get_resized_images(image_path)

        #if in testing mode show the result from fetching images
        if TESTING == 1:
            titles = [char + " - original",char + " - gray_image", char + " - thresholded"]
            imgs = [original_image, gray_image, thresholded_image]
            impress.show_multiple_images(titles, imgs)

        #labeling images
        labeled_image,all_regions = ip.get_labeled_regions(thresholded_image)

        #getting ratios
        all_ratios,mean_size = ip.get_region_ratios(all_regions)

        #getting min max pairs
        min_max_pairs = ip.get_region_bounds(all_ratios,TESTING)

        #overkilling for target regions
        good_regions = ip.get_target_regions(all_regions,min_max_pairs,mean_size)

        if TESTING == 1:
            impress.draw_regions_on_image("ALL_REGIONS",original_image,all_regions)

        if TESTING == 1:
            impress.draw_regions_on_image("GOOD_REGIONS",original_image,good_regions)

        cropped = ip.get_cropped_images(good_regions, thresholded_image)

        if TESTING == 1:
            titles = []
            for i in range(len(cropped)):
                titles.append(str(i))
            impress.show_multiple_images(titles,cropped)
        
startup()

#Training_data_creation.py
#import -- as tdc
