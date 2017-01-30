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

def get_license_plate_image_name(x):
    LPI_name = (con.LICENSE_PLATES_FOLDER +
                con.LICENSE_PLATE_IMAGE_PREFIX +
                con.LICENSE_PLATE_IMAGES[x] +
                con.LICENSE_PLATE_IMAGE_SUFFIX)
    return LPI_name

def Main():

    training_success = kn.try_training()

    if training_success == None:
        print("Training failed. Try running Training_data_creation.py file with testing turned off ('TESTING' flag set to 0)")
        return
    else:
        print("KNN training successfull!")
    
Main()

#Main.py
