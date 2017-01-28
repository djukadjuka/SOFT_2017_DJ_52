#Image_presentation.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import regionprops
from skimage.measure import label
from skimage.exposure import histogram

def show_multiple_images(titles,images):
    for i in range(len(images)):
        cv2.imshow(titles[i],images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_image_with_region(title,image,region):
    img = image.copy()
    bbox = region.bbox
    y = bbox[0]
    x = bbox[1]
    h = bbox[2]
    w = bbox[3]
    cv2.rectangle(img,(x,y),(w,h),(0,0,255,255),2)
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_regions_on_image(title,original_image,regions):
    img = original_image.copy()
    for region in regions:
        bbox = region.bbox
        y = bbox[0]
        x = bbox[1]
        h = bbox[2]
        w = bbox[3]
        cv2.rectangle(img,(x,y),(w,h),(0,0,255,255),2)
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#Image_presentation.py
#import -- impres
