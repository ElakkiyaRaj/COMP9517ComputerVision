import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import imutils
import matplotlib.pyplot as plt

sequence = cv2.VideoCapture('DIC-C2DH-HeLa/Sequence 1/t%3d.tif')
size = (19, 19)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)
counter=0
while sequence.isOpened():
    ce = False
    ret, image = sequence.read()
    if ret == False or cv2.waitKey(40) == 27:
        break
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cl1=gray.copy()

    #pre-processing
    kernel = np.ones((5,5),np.uint8)
    eroded_img= cv2.erode(cl1,kernel,iterations = 1)
    dilated_img = cv2.dilate(eroded_img,kernel,iterations = 1)
    
    bulr_img= cv2.blur(dilated_img,(41,41))
    equalizeHist_img = cv2.equalizeHist(bulr_img)
    _, cl1= cv2.threshold(equalizeHist_img,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)
#    cv2.imwrite('DIC_C2DH_HeLa_segmentation/Method1/Binary_images/DIC_BIN'+str(count)+'.png',cl1)
#    plt.imshow(cl1,cmap="gray")
#    plt.show()
    
    #watershed implementation for getting Nuclie of cells 
    distance = ndi.distance_transform_edt(cl1)
    local_max = peak_local_max(distance, indices=False, labels=cl1, min_distance=30) #43
    markers = ndi.label(local_max)[0]
    labels = watershed(-distance, markers, mask=cl1)
#    plt.imsave('DIC_C2DH_HeLa_segmentation/Method1/Water_segmented/DIC_WS'+str(count)+'.png',cl1)
#    plt.imshow(labels)
#    plt.show()
    counter=0
    gray_copy=gray.copy()
    #putting bounding box
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        contour = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
        contour = imutils.grab_contours(contour)
        contour_max = max(contour, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour_max)
        if cv2.contourArea(contour_max) < 200:
            continue
        else:
            cv2.rectangle(gray, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
            counter += 1
#    cv2.imwrite('DIC_C2DH_HeLa_segmentation/Method1/Bounded_Boxed/DIC_BB'+str(count)+'.png',gray)
    plt.imshow(gray,cmap="gray")
    plt.show()    