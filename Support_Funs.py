import numpy as np
import os
import sys
import cv2

#ispisuje string @title kao naslov
#u obliku:
#==========
#  naslov
#==========
def titlePrint(title):
    strlen = len(title)
    print_equals_line(20+strlen)
    print " "*9,title
    print_equals_line(20+strlen)

#ispis linije znaka jednako za deljenje redova ispisa
def print_equals_line(line_len):
    print "="*line_len

