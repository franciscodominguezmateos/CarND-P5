'''
Created on Feb 14, 2017

@author: Francisco Dominguez
'''
# In python a feature is just a array of numbers
# A feature extractor take a image and return a feature
import numpy as np
import cv2
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self):
        pass
    def extract(self,img):
        pass
class ConcatenateFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.extractors=[]
    def add(self,extractor):
        self.extractors.append(extractor)
    def extract(self,img):
        features=[]
        for extractor in self.extractors:
            feature=extractor.extract(img)
            features.append(feature)
        return np.concatenate(features)          
class HOGFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 0 # Can be 0, 1, 2, or "ALL"
        self.hog_image=None
    def getHOG(self,img):
        return hog(img, orientations=self.orient, 
                   pixels_per_cell=(self.pix_per_cell, self.pix_per_cell), 
                   cells_per_block=(self.cell_per_block, self.cell_per_block), 
                   transform_sqrt=True, 
                   visualise=True, 
                   feature_vector=True)
    def extract(self,img):
        # Call get_hog_features() with vis=False, feature_vec=True
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                features, self.hog_image = self.getHOG(img[:,:,channel])
                hog_features.append(features)
            hog_features = np.ravel(hog_features)        
        else:
            hog_features,self.hog_image = self.getHOG(img[:,:,self.hog_channel])
        return hog_features 
class ColorHistogramFeatureExtractor(FeatureExtractor):
    def __init__(self,nbins=32,bins_range=(0,256)):
        self.nbins=nbins
        self.bins_range=bins_range
    def extract(self,img):
        # Compute the histogram of the color channels separately
        nbins=self.nbins
        bins_range=self.bins_range
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
class ColorSpatialFeatureExtractor(FeatureExtractor):
    def __init__(self,size=(32,32)):
        self.size=size
    def extract(self,img):
        spatial_features = cv2.resize(img, self.size).ravel()
        return spatial_features
