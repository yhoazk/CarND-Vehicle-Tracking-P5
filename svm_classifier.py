#!/usr/bin/env python

import cv2
import  matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC


class SVM_Classifier():
    def __init__(self, type='LinearSVC', *clf_params):
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        if type == 'LinearSVC':
            self.classifier = LinearSVC(clf_params)

    def get_nextWindow(self):
        pass

    def extract(self, img):
        """
        Extracts the features of the image
        :param img: The image from where the features will be extracted
        :return: A group of features
        """
        pass
    def train(self, X,Y, ):
        """
        Trains the classifier
        :param X: The input featrues
        :param Y: The label indicating if the object is present in the image or not
        :return:
        """
        self.classifier.fit(X,Y)

    def predict(self, X):
        return self.predict(X)

    def heat_map(self):
        pass

    def classify(self, img):
        """
        The complete classify pipeline, extracts, slidewindow, predict, heatmap
        Boxing
        :param img:
        :return: Returns the image with the boxes applied and and car object

        """
        return img # TODO: implement
