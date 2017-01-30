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

def load_and_process_LPI(image_name):
    pass

def extract_regions_LPI(LPI_thresh):
    pass

def show_regions_LPI(LPI_original, LPI_regions):
    pass

#Recognition.py
