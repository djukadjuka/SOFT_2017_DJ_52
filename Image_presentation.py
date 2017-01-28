#Image_presentation.py

import numpy as np
import cv2

def show_multiple_images(titles,images):
    for i in range(len(images)):
        cv2.imshow(titles[i],images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Image_presentation.py
#import -- impres
