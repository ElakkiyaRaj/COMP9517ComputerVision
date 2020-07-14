# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:28:54 2020

@author: elakk
"""
from scipy.spatial import distance as dt
from collections import OrderedDict
import numpy as np


class CellsTracker():
    def __init__(self):
        self.cellID = 0
        self.trackedCells = OrderedDict()
        self.parentCell = OrderedDict()
        self.disappeared = OrderedDict()
        #self.trajectory = OrderedDict(list)

    def addCell(self, center):
        #Add path
        self.trackedCells[self.cellID] = center
        self.disappeared[self.cellID] = 0
        self.cellID += 1

    def removeCell(self, cellID):
        #self.trajectory[cellID].append(self.trackedCells[cellID])
        del self.trackedCells[cellID]
        del self.disappeared[cellID]

    def updateTracker(self, centersList):
        centers = np.zeros((len(centersList), 2), dtype="int")

        for (i, center) in enumerate(centersList):
            centers[i] = center
        if len(self.trackedCells) == 0:
            for c in range(0, len(centers)):
                self.addCell(centers[i])
        else:
            cellID = list(self.trackedCells.keys())
            cellCenter = list(self.trackedCells.values())

            Edistance = dt.cdist(np.array(cellCenter), centers)

            rows = Edistance.min(axis=1).argsort()
            cols = Edistance.argmin(axis=1)[rows]

            visitedRows = set()
            visitedCols = set()

            for (row, col) in zip(rows, cols):
                if row in visitedRows or col in visitedCols:
                    continue
                ID = cellID[row]
                #self.trajectory[ID].append(self.trackedCells[ID])
                self.trackedCells[ID] = centers[col]
                visitedRows.add(row)
                visitedCols.add(col)
            
            unVisitedRows = set(
                range(0, Edistance.shape[0])).difference(visitedRows)
            unVisitedCols = set(
                range(0, Edistance.shape[1])).difference(visitedCols)
            if Edistance.shape[0] >= Edistance.shape[1]:
                for row in unVisitedRows:
                    ID = cellID[row]
                    self.disappeared[ID] += 1
                    if self.disappeared[ID] > 3:
                        self.removeCell(ID)
                for col in unVisitedCols:
                    self.addCell(centers[col])
        return self.trackedCells
    
import cv2
import numpy as np
from collections import deque
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import imutils
import time

a = 'PhC-C2DL-PSC'
b = 'DIC-C2DH-HeLa'
#c = 'Fluo-N2DL-HeLa'
#image = cv2.imread('PhC-C2DL-PSC/t012.tif')
#image_copy = image.copy()
#img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sequence = cv2.VideoCapture('PhC-C2DL-PSC/t%3d.tif')
size = (19,19)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)



def create_zeros_array(rows,cols):
    img_O = np.zeros((rows,cols), dtype = np.uint8)
    return img_O

def contrast_stretching(img_I):
    #Creating the image O
    rows, cols = img_I.shape
    img_O = create_zeros_array(rows,cols)
    #Defiing the variables
    a = 0
    b = 255
    c = np.amin(img_I)
    d = np.amax(img_I)
    img_O = (img_I - c) * ((b - a)/(d - c)) + a
    return img_O.astype(np.uint8)
tracker = CellsTracker()

track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
				(127, 127, 255), (255, 0, 255), (255, 127, 255),
				(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]
while sequence.isOpened():
    box=[]
    ce = False
    ret, image = sequence.read()
    if ret == False or cv2.waitKey(40) == 27:
        break
    image_copy = image.copy()
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = contrast_stretching(img)
    img_min = cv2.erode(img, kernel)
    img_max = cv2.dilate(img_min, kernel)
    img_bg = cv2.subtract(img,img_max)
    gaussian = cv2.GaussianBlur(img_bg, (7,7), 0)
    _, thres = cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    #dialate = cv2.dilate(thres, None, iterations=1)
    dialate = thres
    distance = ndi.distance_transform_edt(dialate)
    local_max = peak_local_max(distance, indices=False, labels=dialate, min_distance=5) #43
    markers = ndi.label(local_max)[0]
    labels = watershed(-distance, markers, mask=dialate)
    counter = 0
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(img.shape, dtype="uint8")
        mask[labels == label] = 255
    
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contours)
        center = None
        contour_max = max(contour, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour_max)
        M = cv2.moments(contour_max)
        if M["m00"] and M["m00"]:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.circle(image, center, 1, (0, 0, 255), -1)
            counter += 1
            frame = image
            box.append(center)
    trackedCells = tracker.updateTracker(box)
    for (cellID, cellCenter) in trackedCells.items():
        text = "ID {}".format(cellID)
        cv2.putText(image, text, (cellCenter[0] - 10, cellCenter[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (cellCenter[0], cellCenter[1]), 4, (0, 255, 0), -1)
#        time.sleep(0.1)
    
       
            
            
            
    cv2.putText(image, "Count: {}".format(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow('Image Sequence_ Edges',np.hstack((image,image_copy)))
    time.sleep(0.1)
cv2.destroyAllWindows()
sequence.release() 

