import numpy as np
import os
import sys
import cv2
import Support_Funs as sf


def Main():
    sf.titlePrint("testing_external_functions")
    img = cv2.imread("google_plate.bmp")
    sf.preprocessImage(img)
    
Main()


