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

try_training()

#KNN.py
