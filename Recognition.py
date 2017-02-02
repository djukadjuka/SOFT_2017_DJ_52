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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    #kernel = np.ones((1,1),
    #                 np.uint8)

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

def PATCH_region_ratios(LPI_regions,TEST_FLAG):
    all_ratios = []
    all_areas = []
    for region in LPI_regions:

        bbox = region.bbox

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        ratio = float(h)/w
        area = float(h*w)
        all_ratios.append(ratio)
        all_areas.append(area)
        
    bns = np.linspace(0,20,41)
    n,bins,patches = plt.hist(all_ratios,bins=bns)
    if TEST_FLAG == 1:
        plt.show()
        print(n)

    bns = np.linspace(0,1000)
    n,bins,patches = plt.hist(all_areas,bins=bns)
    if TEST_FLAG == 1:
        plt.show()
        print(n)

def prep_for_recognition(LPI_regions,LPI_thresh,LPI_original,TEST_FLAG):
    cropped_region_images = []
    flat_images = []

    PATCH_region_ratios(LPI_regions,TEST_FLAG)
    
    #cropped out all regions
    cropped_region_images = ip.get_cropped_images(LPI_regions,LPI_thresh)

    #just to show all cropped images
    if TEST_FLAG == 1:
        titles = []
        for i in range(len(cropped_region_images)):
            titles.append(str(i))
        impress.show_multiple_images(titles,cropped_region_images)

    for cropped_image in cropped_region_images:
        flat_images.append(ip.flatten_image(cropped_image))

    if TEST_FLAG == 1:
        titles = []
        for i in range(len(flat_images)):
            titles.append(str(i))
        impress.show_multiple_images(titles,flat_images)

    return cropped_region_images,flat_images

def map_crops_with_flats(cropped_images,flat_images,TEST_FLAG):
    crop_flat_map = {}

    for i in range(len(cropped_images)):
        crop_flat_pair = []
        crop_flat_pair.append(cropped_images[i])
        crop_flat_pair.append(flat_images[i])
        crop_flat_map[i] = crop_flat_pair

    return crop_flat_map

#presents a POSSIBLE character in the plate
#object should contain this information in this order
#object[0] -> the supposed character
#object[1] -> the supposed character in integer format
#object[2] -> the distance from the nearest neighbour
#object[3] -> the cropped image
#object[4] -> the flat image
class character_in_plate(object):

    def __init__(self, obj):
        self.plate_char = obj[0]
        self.plate_char_int = obj[1]
        self.KNN_distance = obj[2]
        self.cropped_image = obj[3]
        self.flat_image = obj[4]

    def get_char():
        return self.plate_char
    def get_char_int():
        return self.plate_char_int
    def get_KNN_distance():
        return self.KNN_distance
    def get_cropped_image():
        return self.cropped_image
    def get_flat_image():
        return self.flat_image

def form_char_list_by_flats(crop_flat_map):

    object_list = []

    for key in crop_flat_map:
        info_list = []
        cropped_image = crop_flat_map[key][0]   #extract cropped_image
        flat_image = crop_flat_map[key][1]      #extract flat_image

        ret, res, neig, dist = kn.KNN.findNearest(flat_image,10)
        print("<-----------> KEY : [",key,"] <----------->")
        print("MAPPED DISTANCES : ")
        char_map = {}
        for i in range(len(dist[0])):
            char_map[chr(int(neig[0][i]))] = 0

        valid_ranges = 0
            
        for i in range(len(dist[0])):
            char_map[chr(int(neig[0][i]))] += 1
            if int(dist[0][i]) < 6000000:
                valid_ranges+=1
                print("\t [",int(dist[0][i]),"]\t-> ",neig[0][i],"\t-> ", chr(int(neig[0][i])))

        if valid_ranges > 0:
            for key in char_map:
                if char_map[key] > 1:
                    print("\t Character [",key,"]\tappears ",char_map[key],"\t times.")
                    cv2.imshow(chr(int(ret)),cropped_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    

#Recognition.py
