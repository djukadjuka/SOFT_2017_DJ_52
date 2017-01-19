import numpy as np
import os
import sys
import cv2
import Constants as const

#ispisuje string @title kao naslov
#u obliku:
#==========
#  naslov
#==========
def titlePrint(title):
    strlen = len(title)
    printEqualsLine(20+strlen)
    print(" "*9,title)
    printEqualsLine(20+strlen)

#ispis linije znaka jednako za deljenje redova ispisa
def printEqualsLine(line_len):
    print ("="*line_len)

#proces pravljenja grayscale slike contrasta itd..
def preprocessImage(img):

    #gauss adaptive threshold trazi grayscale sliku
    grayscale = gray(img)

    #da gauss napravi bolju sliku maksimizira se kontrast na grayscale slici
    maxContrast = contrast(grayscale)
    h,w = grayscale.shape

    #stavljanje blura na sivu contrast sliku
    blurred = np.zeros((h,w,1),np.uint8)
    blurred = cv2.GaussianBlur(maxContrast,const.GAUSS_SMOOTH,0)

    #pravljenje crno bele slike sa izrazenim granicama svakog
    #elementa slike (uokvirena slova, tablica..)
    threshed = cv2.adaptiveThreshold(blurred,\
                                     255.0,\
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                     cv2.THRESH_BINARY_INV,\
                                     const.ADAPTIVE_THRESH_BLOCK_SIZE,\
                                     const.ADAPTIVE_THRESH_WEIGHT)

    #provera
    #drawImage(threshed)
    return grayscale,threshed
    

#pravljenje grayscale
def gray(img):
    #preuzimanje dimenzija slike
    h,w,s = img.shape

    #kreiranje matrice za hue saturation i value za sliku
    imgHSV = np.zeros((h,w,3),np.uint8)

    #konvertovanje slike u grayscale
    #da bi se posle mogao napraviti visok kontrast
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #preuzimanje parametara hue saturation i value od ~grascale slike
    #i_val predstavlja samu sliku
    i_hue,i_sat,i_val = cv2.split(imgHSV)
    return i_val

#pravljenje kontrasta
def contrast(img):
    #uzimaju se dimenzije grayscale slike
    h,w = img.shape

    #potrebno za eroziju i dilaciju slike
    #da se obrise noise
    imgTop = np.zeros((h,w,1),np.uint8)
    imgBack = np.zeros((h,w,1),np.uint8)

    #pravljenje kernel strukture za eroziju i dilaciju
    #u obliku kvadrata
    struct = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    imgTop = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,struct)
    imgBack = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,struct)
    
    ret0 = cv2.add(img,imgTop)
    ret = cv2.subtract(ret0,imgBack)
    
    return ret


#brzo crtanje slike sa svim podesavanjima
def drawImage(img):
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







