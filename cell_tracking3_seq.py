# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:26:39 2020

@author: elakk
"""

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import imutils
a = 'PhC-C2DL-PSC'
b = 'DIC-C2DH-HeLa'
#c = 'Fluo-N2DL-HeLa' 

sequence = cv2.VideoCapture('PhC-C2DL-PSC/t%3d.tif')
size = (19, 19)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)

while sequence.isOpened():
    ce = False
    ret, image = sequence.read()
    if ret == False or cv2.waitKey(40) == 27:
        break
    image_copy = image.copy()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    #Background Subtraction
    img_min = cv2.erode(img, kernel)
    img_max = cv2.dilate(img_min, kernel)
    img_bg = cv2.subtract(img,img_max)
    #Smoothing and thresholding
    gaussian = cv2.GaussianBlur(img_bg, (7,7), 0)
    _, thres = cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    dialate = cv2.dilate(thres, None, iterations=3)

    #Method 1: Finding contours using water shed
    #Watershed
#    distance = ndi.distance_transform_edt(dialate)
#    local_max = peak_local_max(distance, indices=False, labels=dialate, min_distance=5) #43
#    markers = ndi.label(local_max)[0]
#    labels = watershed(-distance, markers, mask=dialate)
    #Bounding box for method 1    
#    for label in np.unique(labels):
#        if label == 0:
#            continue
#        mask = np.zeros(img.shape, dtype="uint8")
#        mask[labels == label] = 255
#        
#        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#        		cv2.CHAIN_APPROX_SIMPLE)
#        contour = imutils.grab_contours(contour)
#        contour_max = max(contour, key=cv2.contourArea)
#        # Drawing bounding box
#        (x, y, w, h) = cv2.boundingRect(contour_max)
#        #print(cv2.contourArea(contour))
#        if cv2.contourArea(contour_max) < 200:
#            continue
#        else:
#            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#            counter += 1
#   
    
    
    #Method2: Edge detection
    c = np.amin(dialate)
    d = np.amax(dialate)
    edges = cv2.Canny(dialate,c,d)  
    _, contour, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contour, key = cv2.contourArea, reverse = True)
#    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
#    cv2.fillPoly(image, pts =contour, color=(255,255,255))
    
    #Bounding Box for method 2
    counter = 0
    for i in range(len(contours)):
        contour = contours[i]
        if cv2.contourArea(contour) < 200:
            continue
        else:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            counter += 1

    cv2.putText(image, "Count: {}".format(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)


    #cv2.imshow('Image Sequence_ Watershed',np.hstack((image,image_copy)))
    cv2.imshow('Image Sequence_ Edges',np.hstack((image,image_copy)))

    
cv2.destroyAllWindows()
sequence.release() 