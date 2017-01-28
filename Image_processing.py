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
def load_and_get_images(full_path_to_image):                #... (1) tested OK!
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

    #get ordinary thresholded image
    img,t_image = cv2.threshold(gray_image_resized.copy(),
                                127,
                                255,
                                cv2.THRESH_BINARY_INV)
    #create kernel for extended thresholding
    #options
    kernel = np.ones((1,1),
                     np.uint8)

    #open up image to remove small dots and
    #interfierence
    t_image_fix = cv2.morphologyEx(t_image,
                                   cv2.MORPH_OPEN,
                                   kernel)

    return original_image,original_image_resized, gray_image, gray_image_resized, t_image, t_image_fix

#shorthand for getting necessary imgs
def get_resized_images(full_path_to_image):                 #...(1)
    original_image, original_image_r, gray_image, gray_r, thresh, thresh_fix = load_and_get_images(full_path_to_image)
    return original_image_r, gray_r, thresh_fix

#get labeled regions
def get_labeled_regions(thresh_image):                      #...(2)
    labeled_image = label(thresh_image) #just label out the thresholded image
                                        #for extracting regions later
    regions = regionprops(labeled_image)
    return labeled_image,regions

#get image ratios based on height and width
#all training images are black and white
#only the height and width ratios are of interest
#on a resized image of a licence plate
#these ratios will @probably be the same...
def get_region_ratios(all_regions):                      #...(3)

    #list for storing all ratios
    all_ratios = []
    for region in all_regions:

        #get bounding box from current region
        bbox = region.bbox

        #create width and height for region
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        #create the ratio
        ratio = float(h)/w
        all_ratios.append(ratio)
    return all_ratios

#create top and bottom bounds for a region group
def get_region_bounds(all_ratios):                      #...(4)

    #parameters gained after testing
    #and trial and error in previous scraped project
    n,bins,patches = plt.hist(all_ratios,bins=range(0,10,3))
    plt.show()     #-->for showing the histogram for proof
    
    #get the index where the histogram
    #has a maximum y value
    max_bins = np.where(n==n.max())
    min_max_pairs = []
    for b in max_bins[0]:               #because the 0th
                                        #element is an array
        min_max_pair = [bins[b],bins[b+1]]
        min_max_pairs.append(min_max_pair)
    return min_max_pairs

#get the regions that are within the bounds
#because the threshed image may have
#picked up interfierence that was not opened
    #may be overkill....
def get_target_regions(all_regions,min_max_pairs):
    good_regions = []
    for min_max_pair in min_max_pairs:  #the histogram may have
                                        #two same maximum values
        minn = min_max_pair[0]
        maxx = min_max_pair[1]
        for region in all_regions:
            bbox = region.bbox
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            ratio = float(h) / w
            if ratio >= minn and ratio <= maxx:
                good_regions.append(region)
    return good_regions

def get_cropped_images(good_regions_list, thresh_image):
    cropped_images= []

    for region_list in good_regions_list:
        for region in region_list:
            bbox = reg.bbox
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            cropped = thresh_image.copy()[x:w,y:h]
            cropped = cv2.resize(cropped,(con.LETTER_WIDTH,con.LETTER_HEIGHT))
            cropped_images.append(cropped)
    return cropped

#Image_processing.py
#Import -- ip
