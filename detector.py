'''
Created on Feb 14, 2017

@author: Francisco Dominguez
'''
import time
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from feature_extractor import HOGFeatureExtractor
from scipy.ndimage.measurements import label
from windows import WindowsProject


def colorSpace(image,color_space):
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        #elif color_space == 'YCrCb':
        #    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image

class BinaryClassifier:
    def __init__(self,clf=SVC()):
        self.scaler=None
        self.clf=clf
        self.score=None
    def train(self,car_features, notcar_features):
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        print('Feature vector length:', len(X_train[0]))
        # Check the training time for the SVC
        t=time.time()
        self.clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        self.score=self.clf.score(X_test, y_test)
        print('Test Accuracy of SVC = ', round(self.score, 4))
    def predict(self,features):
        #5) Scale extracted features to be fed to classifier
        test_features = self.scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = self.clf.predict(test_features)
        return prediction
    def save(self,fname):
        pickle.dump((self.clf,self.scaler),open(fname,"wb"))
    def load(self,fname):
        self.clf,self.scaler=pickle.load(open(fname, "rb" ))

class Detector():
    def __init__(self,colorSpace="RGB",
                 featureExtractor=HOGFeatureExtractor(),
                 clf=BinaryClassifier(),
                 windowFactory=WindowsProject()):
        self.colorSpace=colorSpace
        self.featureExtractor=featureExtractor
        self.windowFactory=windowFactory
        self.clf=clf
        self.onWindows = []
        self.heatMap=None
        self.heatMapThred=None
        self.hitThreshold=0
    def extractFeatures(self,fileNames):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for fName in fileNames:
            #print(fName)
            # Read in each one by one
            imgBGR = cv2.imread(fName) #read BGR
            imgRGB = cv2.cvtColor(imgBGR,cv2.COLOR_BGR2RGB) #convert to RGB
            # Change color space
            image=colorSpace(imgRGB,self.colorSpace)
            # extract features
            feature=self.featureExtractor.extract(image)
            # Append the new feature vector to the features list
            features.append(feature)
        # Return list of feature vectors
        return features
    def train(self,carFileNames,nocarFileNames):
        carFeatures  =self.extractFeatures(  carFileNames)
        noCarFeatures=self.extractFeatures(nocarFileNames)
        self.clf.train(carFeatures,noCarFeatures)
    def inImageWindow(self,window,img):
        rows=img.shape[0]
        cols=img.shape[1]
        if window[0][1]<0 or window[0][0]<0:
            return False
        if window[1][1]>rows:
            return False
        if window[1][0]>cols:
            return False
        return True    
    def detectWindows(self,imgI):
        img=colorSpace(imgI,self.colorSpace)
        windows=self.windowFactory.process()
        #1) Create an empty list to receive positive detection windows
        self.onWindows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            if not self.inImageWindow(window,img): 
                continue
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            test_features=self.featureExtractor.extract(test_img)
            #5) Scale extracted features to be fed to classifier
            # DONE in BinaryClassifier test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.clf.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                self.onWindows.append(window)
        #8) Return windows for positive detections
        return self.onWindows
    def findHeatMap(self,image,windows):
        # Iterate through list of bboxes
        heatMap = np.zeros_like(image[:,:,0]).astype(np.uint)
        for box in windows:
            # Add += 1 for all pixels inside each bbox
            heatMap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        self.heatMap=heatMap
        return self.heatMap
    def getHeatMap(self):
        color= cv2.applyColorMap(self.heatMap,cv2.COLORMAP_JET)
        print("color=",color.shape)
        imgU8=np.asarray(color,np.uint8)
        print("imgU8=",imgU8.dtype)
        return imgU8
    def applyHeatMapThreshold(self,threshold):
        clipped=np.clip(self.heatMap-threshold,0,255)
        self.heatMapThred= np.asarray(clipped, np.ubyte)
        return self.heatMapThred
    def windowsInHeatMap(self,heatMap):
        labels=label(heatMap)
        self.onWindows=[]
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            self.onWindows.append(bbox)
        return self.onWindows
    def detect(self,img):
        windows=self.detectWindows(img)
        heatMap=self.findHeatMap(img,windows)
        heatMapThred=self.applyHeatMapThreshold(self.hitThreshold)
        return self.windowsInHeatMap(heatMapThred)
        


