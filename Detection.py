#Detection.py file

#pronalazenje i crtanje kontura
#http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html


import cv2
import numpy as np
import math
import Constants as const
import random
import Training
import Support_Funs as sf

def detectPlates(originalImage):
    plates = [] #moguce tablice pronadjene u slici

    #karakteristike slike
    h,w,channels = originalImage.shape

    #pravljenje grayscale slike i thresholdovanje slike za preuzimanje kontura slike
    grayscale,threshold = sf.preprocessImage(originalImage)

    #provera rezultata grayscaleovanja i thresholdovanja
    sf.drawImages([originalImage,threshold,grayscale],["ORIGINAL","THRESHOLD","GRAYSCALE"])

    #pretraga karaktera u slici
    chars = detectCharacters(threshold)
    
    return plates

def detectCharacters(thresholdedImage):
    characters = [] #pronadjeni karakteri

    #kopiram sliku jer find contours iz nekog razloga promeni sliku
    thresholdedImageCopy = thresholdedImage.copy()

    #pronalazenje svih kontura slike
    #potrebno za pustanje kroz KNN za izdvajanje karaktera
    imageContours,contours,hierrarchy = cv2.findContours(thresholdedImageCopy,
                                                         cv2.RETR_LIST,
                                                         cv2.CHAIN_APPROX_SIMPLE)

    #crtanje svake konture radi provere
    #sf.drawEachContour(contours,thresholdedImage)

    ##############################################################
    ##TODO: PRETRAZI KARAKTERE U SLICI
    ##      PROVERI ZA SVAKU KONTURU DA LI MOZE DA SE KLASIFIKUJE
    ##############################################################
