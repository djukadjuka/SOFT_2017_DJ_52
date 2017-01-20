#Main.py FILE

#Sve sa pip install
import numpy as np
import os
import sys
import cv2

#Pomocne funkcije - sve sto ne mozes da smislis gde bi moglo stavljaj vamo
import Support_Funs as sf

#Za neke konstantne vrednosti
import Constants as const

#Za treniranje neuronske mreze
import Training

#Za prepoznavanje tablica i karaktera
import Detection

def Main():

    #preuzmi neku sliku iz foldera
    img = cv2.imread(const.IMAGE_NAMES[1])  #koristeci konstantu

    #probaj da naucis mrezu koristeci skinute podatke
    knnSuccessful = Training.KNNTraining()

    #ukoliko ista prodje lose ispisi gresku
    if knnSuccessful == False:
        print("KNN was not successful, check previous error codes.")
        return

    #prijavi da je sve OK!
    print("KNN training successful.")

    #ako nije pronasao sliku ....
    if img is None:
        print("Unable to load image file. Try a different image file and try again.")
        return

    #probaj da pronadjes tablice
    plates = Detection.detectPlates(img)
    
Main()


