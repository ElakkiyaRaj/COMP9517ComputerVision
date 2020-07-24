import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import imutils

sequence = cv2.VideoCapture('Fluo-N2DL-HeLa/Sequence 1/t%3d.tif')
size = (29,29)
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
    

##    img_max = cv2.dilate(img, kernel)    
##    img_min = cv2.erode(img_max, kernel)
##    img_bg = cv2.subtract(img,img_min)
    
    
    #cv2.imshow('Image',img_bg)
    #Smoothing and thresholding
    gaussian = cv2.GaussianBlur(img_bg, (7,7), 0)
    _, thres = cv2.threshold(gaussian,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    dilate = cv2.dilate(thres, None, iterations=3)
    #cv2.imshow('Image',dilate)
    
    c = np.amin(dilate)
    d = np.amax(dilate)
    canny_output = cv2.Canny(dilate,c,d)  
    _, contour, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contour, key = cv2.contourArea, reverse = True)
    

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
    
    counter = 0
    box = []
    for i in range(len(contours)):
        contours2 = cv2.findContours(canny_output, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour2 = imutils.grab_contours(contours2)
        centre = None
        contour_max = max(contour, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour_max)
        M = cv2.moments(contour_max)
        if M["m00"] and M["m00"]:
            centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            box.append(centre)
        color = (255, 255, 0)
        cv2.rectangle(dilate, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        counter += 1

    cv2.putText(dilate, "Count: {}".format(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        
    cv2.imshow('Fluo-N2DL-HeLa', dilate)

    
cv2.destroyAllWindows()
sequence.release() 
