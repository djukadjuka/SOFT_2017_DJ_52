#Training.py file

import numpy as np
import cv2
import sys
import os
import matplotlib as ml
import io
import Support_Funs as sf
import Constants as const

IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

def KNNTraining():
    try:
        #ucitaj sve klasifikacije kao floatovi
        #ove klasifikacije su ustv ascii vrednosti karaktera
        #koje se koriste za dalje proveravanje
        #za sada je sitan fajl posto nisam imao nesto specijalno
        #karaktera sa tablica da treniram mrezu
        #do kraja ce se valjda dodati jos
        classifications = np.loadtxt("classifications_djuka.txt",np.float32)    
    except:
        print("Classification data not provided.",
              " Create classification data and try again.")
        return False

    try:
        #@see classes.txt s tim da su flats vrednosti karaktera na SLICI
        flats = np.loadtxt("flat_images_djuka.txt",np.float32)
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

def createTrainingData():
    #sve slicice koje ce se koristiti za treniranje
    imgs = []
    grayscales = []
    thresholds = []
    extractedCharactersMap = {}     #mapa pronadjenih karaktera
    goodContoursMap = {}            #mapa dobrih kontura za karaktere
    numpyFlatImages =  np.empty((0, IMAGE_WIDTH * IMAGE_HEIGHT))
    
    for i in range(len(const.TRAINING_IMAGE_NAMES)):
        pth = ( const.IMAGES_FOLDER +
                const.TRAINING_IMAGE_NAMES[i] +
                const.IMAGES_EXTENSION)

        #samo citanje slike sa putanje
        #ako se dobije normalna slika (!None) moze da se ubaci u listu
        #i da se radi grayscale i contrast (thresholding)
        testImage = cv2.imread(pth)
        if(testImage is None):
            continue

        imgs.append(testImage)
        gray,thresh = sf.preprocessImage(testImage)
        
        #testiranje ucitavanja i procesiranja slika
        #sf.drawImages([gray,thresh],["gray","thresh"])

        grayscales.append(gray)
        thresholds.append(thresh)

    #za svaku thresholdovanu slicicu trebaju da se
    #nadju konture pa da se zabeleze podaci za sliku
    for i in range(len(thresholds)):
        thImage = thresholds[i].copy()
        
        threshImage,contours,hierrarchy = sf.createContours(thImage)

        #potrebno je naci neku srednju vrednost za sve konture
        #da se ne uzimaju konture koje su jako sitne
        #tipa kvadratici ili kruzici koji su delovi slika
        #ili delimicne cifre ili brojevi
        meanContourArea = 0
        for c in range(len(contours)):
            cArea = cv2.contourArea(contours[c])
            meanContourArea += cArea
        meanContourArea/=len(contours)
        meanContourArea
        
        #sf.drawEachContour(contours,thImage)
        goodContourList = []
        extractedCharacterList = []
        for c in range(len(contours)):
            cArea = cv2.contourArea(contours[c])
            if(cArea > meanContourArea):
                
                #uzmi atribute kvadrata oko konture
                x,y,w,h = cv2.boundingRect(contours[c])
                karakter = thImage[y:y+h,x:x+w]
                karakter = cv2.resize(karakter,(IMAGE_WIDTH,IMAGE_HEIGHT))
                extractedCharacterList.append(karakter)

                #crtanje uokvirenog karaktera
                #boundingImage = imgs[i].copy()
                #cv2.rectangle(boundingImage,(x,y),(x+w,y+h),(0,0,255),1)
                #cv2.imshow("rect",boundingImage)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                #prikazi karakter za proveru
                #cv2.imshow("letter",karakter)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                goodContourList.append(contours[c])
        extractedCharactersMap[const.TRAINING_IMAGE_NAMES[i]] = extractedCharacterList
        goodContoursMap[const.TRAINING_IMAGE_NAMES[i]] = goodContourList

    #prikaz kljuceva u mapi i njigovih
    #integer vrednosti
    #for key in sorted(goodContoursMap):
    #    print(key," -> INT : ",ord(key))
    #provera dal je dobro nasao slike karaktera
    #for key in sorted( extractedCharactersMap ):
    #    for character in extractedCharactersMap[key]:
    #        sf.drawImage(character,key)

    flatImages = []
    classifications = []
    for key in sorted( extractedCharactersMap ):            #za svaki kljuc koji je pronadjen
                                                            #dodaj u listu integer vrednost
        for i in range(len(extractedCharactersMap[key])):   #za svaku sliku/konturu
            image = extractedCharactersMap[key][i]          #preuzmi sliku
                                                            #karaktera
            classifications.append(ord(key))
            #promeni velicinu slike da sve budu iste
            float_image_reshaped = image.reshape((1,IMAGE_WIDTH*IMAGE_HEIGHT))

            #pretvori sliku u float niz
            float_image_reshaped = np.float32(float_image_reshaped)

            #dodaj niz u sve slike
            flatImages.append(float_image_reshaped)
            
            #provera velicine :=> dobijao nes sitno tipa 20 elemenata u nizu
            #sad se dobija >=~400
            #print(len(float_image_reshaped[0]))
            #sf.drawImage(float_image_reshaped,"20x30")
            numpyFlatImages= np.append(numpyFlatImages,float_image_reshaped,0)

    floatClassifications = np.array(classifications,np.float32)
    npArr = floatClassifications.reshape((floatClassifications.size,1))

    np.savetxt("classifications_djuka.txt",npArr)
    np.savetxt("flat_images_djuka.txt",numpyFlatImages)

    print("Training data created. -> [classifications_djuka.txt,flat_images_djuka.txt]")

#
#createTrainingData()
