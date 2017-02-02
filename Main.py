#Main.py

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
import Recognition as rec

TESTING = 0

def get_license_plate_image_name(x):
    LPI_name = (con.LICENSE_PLATES_FOLDER +
                con.LICENSE_PLATE_IMAGE_PREFIX +
                con.LICENSE_PLATE_IMAGES[x] +
                con.LICENSE_PLATE_IMAGE_SUFFIX)
    return LPI_name

def get_new_LPI_name(x):
    LPI_NAME = (con.NEW_PLATES_FOLDER +
                con.NEW_PLATE_PREFIX +
                con.NEW_PLATE_NAMES[x] +
                con.NEW_PLATE_SUFFIX)
    return LPI_NAME

#display flag to show specific images
#   0   -   Show no images
#   1   -   Show all images
#   2   -   Show resized images
#   3   -   Show original images
def get_images(LPI_name, display_flag = 0):
    LPI_original, LPI_original_resized, LPI_gray, LPI_gray_resized, LPI_thresh, LPI_thresh_resized = rec.load_and_process_LPI(LPI_name)

    images = None
    titles = None
    if display_flag == 1 and TESTING == 1:
        images = [LPI_original, LPI_original_resized, LPI_gray, LPI_gray_resized, LPI_thresh, LPI_thresh_resized]
        titles = ["Original","Original Resized","Grayscale","Grayscale resized","Threshold","Threshold resized"]
    elif display_flag == 2 and TESTING == 1:
        images = [LPI_original_resized, LPI_gray_resized, LPI_thresh_resized]
        titles = ["Original Resized","Grayscale resized","Threshold resized"]
    elif display_flag == 3 and TESTING == 1:
        images = [LPI_original, LPI_gray, LPI_thresh]
        titles = ["Original","Grayscale","Threshold"]

    if images is not None:
        impress.show_multiple_images(titles,images)

    return LPI_original,LPI_original_resized, LPI_gray, LPI_gray_resized, LPI_thresh, LPI_thresh_resized
    
def Main():

    training_success = kn.try_training()

    if training_success == None:
        print("Training failed. Try running Training_data_creation.py file with testing turned off ('TESTING' flag set to 0)")
        return
    else:
        print("KNN training successfull!")

    #LPI_name = get_license_plate_image_name(1)
    LPI_name = get_new_LPI_name(2)
    LPI_original, LPI_original_resized, LPI_gray, LPI_gray_resized, LPI_thresh, LPI_thresh_resized = get_images(LPI_name,3)
    LPI_labeled, LPI_regions = rec.extract_regions_LPI(LPI_thresh_resized)

    if TESTING == 1:
        rec.show_regions_LPI(LPI_original_resized,LPI_regions)

    cropped_images,flat_images = rec.prep_for_recognition(LPI_regions,LPI_thresh_resized,LPI_original_resized,TESTING)

    crop_flat_map = rec.map_crops_with_flats(cropped_images,flat_images,TESTING)
    
    rec.form_char_list_by_flats(crop_flat_map)
    
Main()

#Main.py
