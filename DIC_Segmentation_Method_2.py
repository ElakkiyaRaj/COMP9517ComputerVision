import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
import imutils
from matplotlib import pyplot as plt

sequence = cv2.VideoCapture('DIC-C2DH-HeLa/Sequence 1/t%3d.tif')
size = (19, 19)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)
counter=0
count=0
while sequence.isOpened():
    ce = False
    ret, img = sequence.read()
    if ret == False or cv2.waitKey(40) == 27:
        break
    """ pre-processing """    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cl1=gray.copy()
    cl2=gray.copy()
    
    kernel = np.ones((7,7),np.uint8)
    cl1 = cv2.erode(cl1,kernel,iterations = 1)
    cl1 = cv2.dilate(cl1,kernel,iterations = 1)
    cl1 = cv2.add(gray,cl1)
    
    cl1 = cv2.equalizeHist(cl1)
    
    gaussian = cv2.GaussianBlur(cl1, (7,9),10)
    _, cl1= cv2.threshold(gaussian ,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    
    cl1 = cv2.erode(cl1,kernel,iterations = 1)
    """ Watershed Segmentation """    
    distance = ndi.distance_transform_edt(cl1)
    local_max = peak_local_max(distance, indices=False, labels=cl1, min_distance=35) #43
    markers = ndi.label(local_max)[0]
    labels = watershed(-distance, markers, mask=cl1)
#    plt.imsave('DIC_C2DH_HeLa_segmentation/Method2/Water_segmented/DIC_WS'+str(count)+'.png',cl1)
#    plt.imshow(labels)
#    plt.show()

    """ Bounding Box"""    
#    counter=0
#    for label in np.unique(labels):
#        if label == 0:
#            continue
#        mask = np.zeros(gray.shape, dtype="uint8")
#        mask[labels == label] = 255
#        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                cv2.CHAIN_APPROX_SIMPLE)
#        contour = imutils.grab_contours(contour)
#        contour_max = max(contour, key=cv2.contourArea)
#        # Drawing bounding box
#        (x, y, w, h) = cv2.boundingRect(contour_max)
#        #print(cv2.contourArea(contour))
#        if cv2.contourArea(contour_max) < 200:
#            continue
#        else:
#            cv2.rectangle(cl2, (x, y), (x+w, y+h), (0, 255, 0), 2)
#            counter += 1
#    plt.imshow(cl2,cmap="gray")
#    plt.show()