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


#save all extracted training data
#to default files for classifications
#and flat images
def save_data(np_float_classifications_array,np_flat_images_array):
    np.savetxt(con.CLASSIFICATIONS_FILE,np_float_classifications_array)
    np.savetxt(con.FLAT_IMAGES_FILE,np_flat_images_array)

def startup():

    flat_images_map = {}

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

        #getting cropped images
        cropped_images = ip.get_cropped_images(good_regions, thresholded_image)

        if TESTING == 1:
            titles = []
            for i in range(len(cropped_images)):
                titles.append(str(i))
            impress.show_multiple_images(titles,cropped_images)

        #forming cropped images as flat images
        flat_crops = []
        for crop in cropped_images:
            flat_crops.append(ip.flatten_image(crop))

        if TESTING == 1:
            titles = []
            for i in range(len(flat_crops)):
                titles.append(str(i))
            impress.show_multiple_images(titles,flat_crops)

        #mapping cropped images
        flat_images_map[char] = flat_crops
    
    classifications_list = []
    #print(con.LETTER_WIDTH*con.LETTER_HEIGHT)
    numpy_flat_images = np.empty((0,con.LETTER_WIDTH*con.LETTER_HEIGHT))
    for key in flat_images_map:
        
        if TESTING == 1:
            print("{",key,"} contains :", len(flat_images_map[key])," images.")    
        for crop in flat_images_map[key]:
            classifications_list.append(ord(key))
            numpy_flat_images = np.append(numpy_flat_images,crop[0])

        if TESTING == 1:
                print(len(numpy_flat_images))

    #flatten the classification list
    flat_classifications = np.array(classifications_list,np.float32)
    flat_classifications = flat_classifications.reshape((flat_classifications.size,1))

    #save the data
    save_data(flat_classifications, numpy_flat_images)

    
startup()

#Training_data_creation.py
#import -- as tdc
