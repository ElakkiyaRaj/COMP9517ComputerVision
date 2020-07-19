# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:28:54 2020

@author: elakk

Completed integrating task 1 and task 3

Compile the code by typing the following in the terminal.

Task 1:
------
python PhC-C2DL-PSC_Kalman_revision3.py --seq PhC-C2DL-PSC --dt 8 --sv 0.5 --mv 1 --bias 100 --max 0 --task 1

Task 2:
------
python PhC-C2DL-PSC_Kalman_revision3.py --seq PhC-C2DL-PSC --dt 8 --sv 0.5 --mv 1 --bias 100 --max 0 --task 2

Task 3:
------
python PhC-C2DL-PSC_Kalman_revision3.py --seq PhC-C2DL-PSC --dt 8 --sv 0.5 --mv 1 --bias 100 --max 0 --task 3

"""
import argparse
from collections import OrderedDict
import numpy as np
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import imutils

""" """


def create_zeros_array(rows, cols):
    """ Initialise a zero array"""
    img_O = np.zeros((rows, cols), dtype=np.uint8)
    return img_O


def contrast_stretching(img_I):
    """ Perform constrast strectching"""
    rows, cols = img_I.shape
    img_O = create_zeros_array(rows, cols)
    a = 0
    b = 255
    c = np.amin(img_I)
    d = np.amax(img_I)
    img_O = (img_I - c) * ((b - a)/(d - c)) + a
    return img_O.astype(np.uint8)


class KalmanFilter(object):

    def __init__(self, center, dt=8, sv=6, mv=1):
        """ Initialise Kalman filer """
        super(KalmanFilter, self).__init__()
        self.stateVariance = sv  # E
        self.measurementVariance = mv
        self.dt = dt

        # Vector of observation
        self.b = np.array([[0], [255]])
        # A - state transition matrix
        self.A = np.matrix([[1, self.dt, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dt],  [0, 0, 0, 1]])
        # observation covariance matix- constant throughout the state
        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])  # [0, 0, 1, 0]

        # Error covariance = E*State identity matrix
        self.errorCov = np.matrix(
            self.stateVariance*np.identity(self.A.shape[0]))
        # Observation noise matrix
        self.R = np.matrix(self.measurementVariance*np.identity(
            self.H.shape[0]))
        # Process noise matrix
        self.Q = np.matrix([[self.dt**3/3, self.dt**2/2, 0, 0],
                            [self.dt**2/2, self.dt, 0, 0],
                            [0, 0, self.dt**3/3, self.dt**2/2],
                            [0, 0, self.dt**2/2, self.dt]])

        # self.Q = np.matrix([[self.dt**4/4, self.dt**3/2, 0, 0],
        #                     [self.dt**3/2, self.dt**2, 0, 0],
        #                     [0, 0, self.dt**4/4, self.dt**3/2],
        #                     [0, 0, self.dt**3/2, self.dt**2]])

        # Current state of the cell
        self.state = np.matrix([[0], [1], [0], [1]])
        # Predicted state
        self.predictedstate = None

    def predict(self):
        """Predicts the next state of the cell using the 
            previous state information"""
        # X(i) = A*X(i-1)
        self.state = self.A*self.state
        # P(i) = A*P(i-1)*A(Transpose) + Q
        self.predictedErrorCov = self.A*self.errorCov*self.A.T + self.Q
        state_array = np.asarray(self.state)
        self.predictedstate = state_array[0], state_array[2]
        return state_array[0], state_array[2]

    def correct(self, center, flag):
        """Updates the predicted state using the current measurement"""
        if not flag:
            temp = np.asarray(self.state)
            self.b = [temp[0], temp[2]]
        else:
            self.b = center
        # K(i) = P(i)*H(Transpose) * Inverse of ( H*P(i)*H(Transpose) + R )
        self.kalmanGain = self.predictedErrorCov*self.H.T*np.linalg.pinv(
            self.H*self.predictedErrorCov*self.H.T+self.R)
        # X(i) = X(i) + K(i) * (y(i) - H*X(i))
        self.state = self.state + self.kalmanGain * \
            (self.b - (self.H*self.state))
        # P(i) = (I - K(i)*H)* P(i)
        self.erroCov = (np.identity(self.errorCov.shape[0]) -
                        self.kalmanGain*self.H)*self.predictedErrorCov

        state_array = np.asarray(self.state)
        self.predictedstate = state_array[0], state_array[2]
        return self.state, np.array(self.predictedstate).reshape(1, 2)


class Paths():
    def __int__(self):
        """ Initialise Kalman filter objects """
        self.cellID = None
        self.predictedState = None

    def addPath(self, center, cellID):
        """  Assign kalman filter object for all the cells"""
        self.trajectory = deque()
        self.KF = KalmanFilter(center, args.dt, args.sv, args.mv)
        self.missedFrame = 0
        self.predictedState = center.reshape(1, 2)
        self.cellID = cellID

    def updatePrediction(self, center):
        """ Predict the state of the cell """
        self.predictedState = np.array(self.KF.predict()).reshape(1, 2)
        self.KF.correct(np.matrix(center).reshape(2, 1))


class CellsTracker():
    def __init__(self, bias, max_missedFrame):
        """ Intialise the tracker object """
        self.cellID = 0
        self.bias = bias
        self.max_missedFrame = max_missedFrame
        self.trackedCells = []
        self.parentCell = OrderedDict()
        self.disappeared = OrderedDict()
        self.currentTracker = []

    def addCell(self, center, cellID):
        """ Update / Add tracker objects to the tracker """
        pathObj = Paths()
        pathObj.addPath(center, cellID)
        self.trackedCells.append(pathObj)
        self.currentTracker.append(cellID)

    def removeCell(self, cellID):
        """ Delete objects from tracker"""
        del self.trackedCells[cellID]

    def updateTracker(self, centersList):
        """ Update the state information of objects in the tracker """
        centers = np.zeros((len(centersList), 2), dtype="int")
        # Assinging individual trackers to the cells
        for (i, center) in enumerate(centersList):
            centers[i] = center
        if len(self.trackedCells) == 0:
            for c in range(0, len(centers)):
                self.addCell(centers[c], c)

        # Updating the Frobenius cost norm
        cost = np.zeros(shape=(len(self.trackedCells), len(centers)))
        for i in range(len(self.trackedCells)):
            for j in range(len(centers)):
                try:
                    temp = self.trackedCells[i].predictedState - centers[j]
                    Fnorm = np.linalg.norm(temp, axis=1)
                    cost[i][j] = Fnorm
                except:
                    pass

        # Applying the Hungarian min weight matching algorithm
        cost = cost * 0.5
        row, col = linear_sum_assignment(cost)

        # Updating the min-weight matched pairs
        visitedPath = [-1]*len(self.trackedCells)
        for i in range(len(row)):
            visitedPath[row[i]] = col[i]

        unVisitedPath = []
        # Checking for the distance bias
        for i in range(len(visitedPath)):
            if visitedPath[i] != -1:
                if cost[i][visitedPath[i]] > self.bias:
                    visitedPath[i] = -1
                    unVisitedPath.append(i)
            else:
                self.trackedCells[i].missedFrame += 1

        lost_frame = []
        # Checking for lost cells
        for i in range(len(self.trackedCells)):
            if self.trackedCells[i].missedFrame > self.max_missedFrame:
                lost_frame.append(i)

        # Delete the record of lost cells from the tracker
        lost_frame = sorted(lost_frame, reverse=True)
        if len(lost_frame) > 0:
            for i in range(len(lost_frame)):
                self.removeCell(lost_frame[i])
                del visitedPath[lost_frame[i]]
                del self.currentTracker[lost_frame[i]]

        # Validate if all the current states are associated with a tracker object
        unassigned = []
        for i in range(len(centers)):
            if i not in visitedPath:
                unassigned.append(i)

        # Assign trackers for new cells in the current frame
        if len(unassigned) != 0:
            for i in range(len(unassigned)):
                if unassigned[i] not in self.currentTracker:
                    self.addCell(centers[unassigned[i]], unassigned[i])

        # Updating the prediction matrix based on the previous state information
        # Store the path of the cells based on the predicted state
        for i in range(len(visitedPath)):
            c = np.matrix([[0], [1]])
            self.trackedCells[i].KF.predict()
            if (visitedPath[i] != -1):
                self.trackedCells[i].missedFrame = 0
                self.trackedCells[i].KF.state, self.trackedCells[i].predictedState = self.trackedCells[i].KF.correct(
                    np.matrix(centers[visitedPath[i]]).reshape(2, 1), 1)
            else:
                self.trackedCells[i].KF.state, self.trackedCells[i].predictedState = self.trackedCells[i].KF.correct(
                    np.matrix(c).reshape(2, 1), 0)
            self.trackedCells[i].trajectory.append(
                self.trackedCells[i].predictedState)


if __name__ == '__main__':
    print("Connecting....")
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default='PhC-C2DL-PSC',
                        help='PhC-C2DL-PSC, DIC-C2DH-HeLa or Fluo-N2DL-HeLa')
    parser.add_argument('--dt', type=int, default=8,
                        help='Time delta')
    parser.add_argument('--sv', type=float, default=0.5, help='State Variance')
    parser.add_argument('--mv', type=int, default=1,
                        help='Measurement variance')  # 10
    parser.add_argument('--task', type=int, default=1,
                        help='Task number')
    parser.add_argument('--bias', type=int, default=100,
                        help='Distance bias for cost function')  # 10
    parser.add_argument('--max', type=int, default=0,
                        help='Maximum value of missed frame')
    args = parser.parse_args()
    print("Reading file", args.seq+'/t%3d.tif')
    sequence = cv2.VideoCapture(args.seq+'/t%3d.tif')
    size = (19, 19)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    tracker = CellsTracker(args.bias, args.max)

    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0),
              (127, 127, 255), (255, 0, 255), (255, 127, 255),
              (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]

    while sequence.isOpened():
        box = []
        ret, image = sequence.read()
        if ret == False or cv2.waitKey(40) == 27:
            break
        image_copy = image.copy()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Pre-processing of the PhC-C2DL-PSC cells
        if args.seq == 'PhC-C2DL-PSC':
            img = contrast_stretching(img)
            img_min = cv2.erode(img, kernel)
            img_max = cv2.dilate(img_min, kernel)
            img_bg = cv2.subtract(img, img_max)
            gaussian = cv2.GaussianBlur(img_bg, (7, 7), 0)
            _, thres = cv2.threshold(
                gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            #dialate = cv2.dilate(thres, None, iterations=1)
            dialate = thres
            distance = ndi.distance_transform_edt(dialate)
            local_max = peak_local_max(
                distance, indices=False, labels=dialate, min_distance=5)  # 43
            markers = ndi.label(local_max)[0]
            labels = watershed(-distance, markers, mask=dialate)

        elif args.seq == 'DIC-C2DH-HeLa':
            """ <Kamal : To add the code block here>
                1. Perform pre-processing of the 'DIC-C2DH-HeLa cells
                2. Segment the cells either using edge contour or watershed
                3. label the segmented cell and return it as "labels" variable
                Note : Tracker might not work as expected for this image sequence with the current initialisation vector.
                Might have to adjust the meta parameters. (args.dt, args.sv..)
                """
            pass
        else:
            """ <Sunil: To add the code block here>
                1. Perform pre-processing of the 'Fluo-N2DL-HeLa cells
                2. Segment the cells either using edge contour or watershed
                3. label the segmented cell and return it as "labels" variable
                Note : Tracker might not work as expected for this image sequence with the current initialisation vector.
                Might have to adjust the meta parameters. (args.dt, args.sv..)
                """
            pass

        # Track the segmented cells
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
                counter += 1
                frame = image
                box.append(center)
        tracker.updateTracker(box)

        # Task 1: Displays the trajectory of the cells
        if args.task == 1:
            for i in range(len(tracker.trackedCells)):
                no = round(i % 10)
                if (len(tracker.trackedCells[i].trajectory) > 1):
                    x = int(tracker.trackedCells[i].trajectory[-1][0, 0])
                    y = int(tracker.trackedCells[i].trajectory[-1][0, 1])
                    tl = (x-10, y-10)
                    br = (x+10, y+10)
                    cv2.rectangle(image, tl, br, (0, 255, 0), 1)

                    for k in range(1, len(tracker.trackedCells[i].trajectory) - 1):
                        x = int(tracker.trackedCells[i].trajectory[k][0, 0])
                        y = int(tracker.trackedCells[i].trajectory[k][0, 1])
                        cv2.circle(image_copy, (x, y), 1, colors[no], -1)
                    cv2.circle(image_copy, (x, y), 2, colors[no], -1)
            cv2.putText(image, "Count: {}".format(counter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (40, 24, 50), 2)
            cv2.imshow('Image Sequence_ Edges', np.hstack((image, image_copy)))

        # Task 2: Detect cell division
        elif args.task == 2:
            """ <Vishal: To add the code block here>
                1. Identify the cells under mitosis
                2. With task 1 as reference , draw bounding boxes around the cells
                3. Differtiate the dividing cells (with diff colour bounding box)
                4. Display the count 
                """
            pass

        # Task 3: Perform track measurement
        elif args.task == 3:
            bbox = cv2.selectROI("Select ROI", image)
            bbox = np.array(bbox)
            if np.sum(bbox) != 0:
                centroids = []
                for i in range(len(tracker.trackedCells)):
                    no = round(i % 10)
                    if (len(tracker.trackedCells[i].trajectory) > 1):
                        x = int(tracker.trackedCells[i].trajectory[-1][0, 0])
                        y = int(tracker.trackedCells[i].trajectory[-1][0, 1])

                        if x > bbox[1] and x < bbox[1]+bbox[3] and y < bbox[0] and y < bbox[0]+bbox[2]:
                            for k in range(1, len(tracker.trackedCells[i].trajectory) - 1):
                                x = int(
                                    tracker.trackedCells[i].trajectory[k][0, 0])
                                y = int(
                                    tracker.trackedCells[i].trajectory[k][0, 1])
                                roi = image[bbox[1]:bbox[1] +
                                            bbox[3], bbox[0]: bbox[0]+bbox[2]]
                                centroids.append([x, y])
                            if len(centroids) > 1:
                                break

                if len(centroids) > 1:
                    TotalDistance = 0
                    speed = spatial.distance.cdist(np.array(
                        centroids[-1]).reshape(1, 2), np.array(centroids[-2]).reshape(1, 2))[0][0]
                    for i in range(len(centroids) - 1):
                        TotalDistance += spatial.distance.cdist(np.array(centroids[i]).reshape(
                            1, 2), np.array(centroids[i+1]).reshape(1, 2))[0][0]
                    NetDistance = spatial.distance.cdist(np.array(centroids[0]).reshape(
                        1, 2), np.array(centroids[-1]).reshape(1, 2))[0][0]
                    CRatio = TotalDistance / NetDistance
                    print(
                        "------------------------------------------------------------")
                    print(
                        'Speed of the cell is: {:.2f} pixels/frame'.format(speed))
                    print('Total distance of the cell is: {:.2f} pixels'.format(
                        TotalDistance))
                    print('Net distance of the cell is: {:.2f} pixels'.format(
                        NetDistance))
                    print('Cratio of the cell is: {:.2f}'.format(CRatio))
                    print(
                        "------------------------------------------------------------")
            else:
                continue

    cv2.destroyAllWindows()
    sequence.release()
