import numpy as np
import cv2
import sys
import os
import matplotlib as ml
import io
import Support_Funs as sf

def pyDataToFile(filename,data):
    file = open(filename,"a")
    file.write(data)
    file.close()
    
def fileDataToPy(filename):
    file = open(filename,'r')
    data = file.read()
    print(data)
    file.close()
    return data

img = cv2.imread("google_plate.bmp")

gray,thresh = sf.preprocessImage(img)

sf.drawImage(img)
sf.drawImage(thresh)
sf.drawImage(gray)
