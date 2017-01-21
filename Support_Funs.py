#Support_Funs.py FILE

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
def drawImage(img,title):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#brzo crtanje vise slika koje se gase pritiskom bilo kog tastera
#velicina liste slika i velicine liste naslova mora biti ista
def drawImages(images,titles):
    for i in range(len(images)):
        cv2.imshow(titles[i],images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#crtanje iz drugog foldera
#posto mora prvo da iscita slicicu pa onda
#da je iscrta ako nema direktnu putanju
def drawImageFromFolder(path):
    img = cv2.imread(path)
    drawImage(img,path)

#funkcija crta jednu prosledjenu konturu
def drawOneContour(contour,imageCopy):
        #napravi masku nula
        mask = np.zeros(imageCopy.shape,np.uint8)

        cv2.drawContours(mask,contour,-1,(255,255,255,255),3)

        removed = cv2.bitwise_and(imageCopy,mask)

        drawImage(removed,"Contour")
        
#iscrtavanje SVAKE konture jedne slike
#iscrtava posebno konturu po konturu i prikazuje sliku
#pozovi za proveru svaki jedan put jer traje ako slika ima puno kontura
#probaj da prepravis thresholding kako bi se smanjio broj kontura
def drawEachContour(contours,thresholdedImage):

    #iscrtavaj od najvece konture do najmanje
    #koristi samo za proveru pa kad stigne do manjih kontura
    #(tackice i kvadratici koji ne znace nista) sa ctrl+c prekini
    #posto ce crtati zauvek ako nisi prepravio thresholding
    for i in range(len(contours)-1,0,-1):
        
        #prekopiraj sliku da ti je ne kvari
        img = thresholdedImage.copy()

        #napravi matru svih nula posle za bitwise andovanje
        mask = np.zeros(img.shape,np.uint8)

        #sortiraj po velicini, mogao bi prekinuti petlju na tipa pola velicine
        #kontura posto ce karakteri sigurno biti medju najvecim
        cnt = sorted(contours,key=cv2.contourArea)

        #iscrtaj konture po masci
        #OVO CE SAMO DA PREPRAVI MASKU NECE NISTA ISCRTATI NA EKRAN
        #CIMANJE 3h ZASTO NIJE CRTAO!!!! 
        cv2.drawContours(mask,cnt,i,(255,255,255,255),3)

        #anduj sta si dobio u masci sa originalom
        #tacno ce tamo gde su kecevi ostaviti konturu
        #ostalo ce obojiti crnim
        #tu sliku sacuvas u removed
        removed = cv2.bitwise_and(img,mask)

        #prikazi tu sliku
        drawImage(removed,"Contour")

#automatsko pravljenje kontura
#da ne moram da pamtim kako se ovo koristi
def createContours(imageCopy):
    imageContours,contours,hierrarchy = cv2.findContours(imageCopy,
                                                         cv2.RETR_LIST,
                                                         cv2.CHAIN_APPROX_SIMPLE)
    return imageContours,contours,hierrarchy
