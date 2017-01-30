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

#reading text files for training
def try_training():

    #reading classifications file
    try:
        classifications = np.loadtxt(con.CLASSIFICATIONS_FILE)
    except:
        print("Classifications file not found/not loaded properly. Check files and try again.")
        return None

    #reading training images file
    try:
        flat_images = np.loadtxt(con.FLAT_IMAGES_FILE)
    except:
        print("Training file not found/not loaded properly. Check files and try again.")
        return None

    #reshaping classifications to a column
    #needs to be a column for KNN object training
    classifications = classifications.reshape((classifications.size,1))

    #convert to ndarray of floats 32
    flat_images = np.asarray(flat_images,dtype="float32")
    classifications = np.asarray(classifications,dtype="float32")

    #set default return neighbour
    KNN.setDefaultK(1)

    #train the KNN object
    #   TRAINING DATA,  SAMPLING BY ROW,    CLASSIFICATION DATA
    KNN.train(flat_images,cv2.ml.ROW_SAMPLE,classifications)    
    return 1

#testing out basic recognition on single
#preprocessed fixed image
def try_recognition():
    #name of image to try and recognize
    test_image = "TRAINING_Q.jpg"

    #START OF IMAGE PROCESSING
    original, original_r, gray, gray_r, thresh, thresh_fix = ip.load_and_get_images(test_image)
    
    labeled_image,all_regions = ip.get_labeled_regions(thresh_fix)
    ratios,mean_size  = ip.get_region_ratios(all_regions)
    min_max_pairs = ip.get_region_bounds(ratios,0)
    good_regions = ip.get_target_regions(all_regions,min_max_pairs,mean_size)
    cropped_images = ip.get_cropped_images(good_regions,thresh_fix)
    #END IMAGE PROCESSING

    #check processed image
    cv2.imshow("crop",cropped_images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #flatteining out images
    flat_images = []
    for img in cropped_images:
        flat = ip.flatten_image(img)

        flat = np.asarray(flat,dtype="float32")
        flat_images.append(flat)
        
        cv2.imshow("crop",flat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #ret -> single returned nearest neighbour
    #res -> listed neighbours
    #neig -> all neighbours
    #dist -> distances of neighbours
    ret, res, neig, dist = KNN.findNearest(flat_images[0],5)
    print(ret,res,neig,dist)
    print(chr(int(ret)))

#KNN.py
