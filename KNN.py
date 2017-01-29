#KNN.py

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

#Knearest object
KNN = cv2.ml.KNearest_create()

def try_training():
    classifications = np.loadtxt(con.CLASSIFICATIONS_FILE)
    flat_images = np.loadtxt(con.FLAT_IMAGES_FILE)
    classifications = classifications.reshape((classifications.size,1))
    
    flat_images = np.asarray(flat_images,dtype="float32")
    classifications = np.asarray(classifications,dtype="float32")
    KNN.setDefaultK(1)
    KNN.train(flat_images,cv2.ml.ROW_SAMPLE,classifications)    

def try_recognition():
    test_image = "TRAINING_TEST.jpg"

    original, original_r, gray, gray_r, thresh, thresh_fix = ip.load_and_get_images(test_image)
    
    labeled_image,all_regions = ip.get_labeled_regions(thresh_fix)
    ratios,mean_size  = ip.get_region_ratios(all_regions)
    min_max_pairs = ip.get_region_bounds(ratios,0)
    good_regions = ip.get_target_regions(all_regions,min_max_pairs,mean_size)
    cropped_images = ip.get_cropped_images(good_regions,thresh_fix)
    cv2.imshow("crop",cropped_images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
try_training()
try_recognition()
#KNN.py
