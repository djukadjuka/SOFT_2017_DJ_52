#Training.py file

import numpy as np
import cv2
import sys
import os
import matplotlib as ml
import io
import Support_Funs as sf

def KNNTraining():
    try:
        #ucitaj sve klasifikacije kao floatovi
        #ove klasifikacije su ustv ascii vrednosti karaktera
        #koje se koriste za dalje proveravanje
        #za sada je sitan fajl posto nisam imao nesto specijalno
        #karaktera sa tablica da treniram mrezu
        #do kraja ce se valjda dodati jos
        classifications = np.loadtxt("classes.txt",np.float32)    
    except:
        print("Classification data not provided.",
              " Create classification data and try again.")
        return False

    try:
        #@see classes.txt s tim da su flats vrednosti karaktera na SLICI
        flats = np.loadtxt("flats.txt",np.float32)
    except:
        print("Flat image file data is not provided.",
              " Create flat image file data and try again.")
        return False

    #prepravi klasifikacije da bude jedna kolona
    classifications = classifications.reshape((
        classifications.size,1))

    
    #pravi se instanca KNearest koji se koristi za treniranje
    KNN = cv2.ml.KNearest_create()

    KNN.setDefaultK(1)

    #treniranje knn klasifikatora
    KNN.train(flats,
              cv2.ml.ROW_SAMPLE,
              classifications)

    return True
