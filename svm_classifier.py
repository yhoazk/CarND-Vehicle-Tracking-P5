#!/usr/bin/env python

import cv2
import  matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from glob import glob
import matplotlib.image as mpimg
from tqdm import tqdm

class SVM_Classifier():
    def __init__(self, type='LinearSVC', *clf_params):
        # show debug info
        self.debug = False
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        if type == 'LinearSVC':
            self.classifier = LinearSVC(clf_params)
        # array to store the windows
        self.windows = []
        #current window
        self.curr_wind = 0
        # stores hog features
        self.hog_array = None
        # defines the search area
        self.image_window = ((70,450),(1210,670))
        self.trained = False
        # Target resize for each window
        self.tgt_resize = (32,32)
        #type of image to train

        #self positive image dataset
        self.feat_p = []
        self.feat_n = []
        self.label_p = []
        self.label_n = []
        self.dataset = {'feat_p':self.feat_p, 'feat_n':self.feat_n, 'lbl_p': self.label_p, 'lbl_n': self.label_n}

    def __calc_windows(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if None in x_start_stop or None in y_start_stop:
            x_start_stop = [0, img.shape[1]]
            y_start_stop = [0, img.shape[0]]
        # Compute the span of the region to be searched
        x_len = x_start_stop[1] - x_start_stop[0]
        y_len = y_start_stop[1] - y_start_stop[0]

        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def __get_nextWindow(self):
        pass

    def __bin_spatial(img, color_space='RGB', size=(32,32)):
        # Convert image to new color space (if specified)
        img_copy = np.copy(img)
        # supposing we already have the image in RGB
        if color_space == 'HSV':
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LUV)

        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img_copy, size).ravel()
        # Return the feature vector
        return features


    def __hog(self, img, orient=9, ppc=8, cpb=2, vis=False, f_vect=False):
        hog_img = None
        if vis:
            self.hog_array, hog_img = hog(img,orient,pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),visualise=vis,feature_vector=f_vect)
        else:
            self.hog_array = hog(img,orient,pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),visualise=vis,feature_vector=f_vect)
        return self.hog_array, hog_img

    def __draw_rect(self,img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy




    def __extract(self, img):
        """
        Extracts the features of the image
        :param img: The image from where the features will be extracted
        :return: A group of features
        """
        # convert to gray for hog
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        feat, img = self.__hog(gray, vis=True)
        if self.debug:
            plt.imshow(img)
            print(feat.shape)
            plt.show()
        return img

    def __extract_images(self, path_p="../vehicles", path_n="../non-vehicles"):
        """
        get thee paths images from the vehicle and non-vehicle datasets
        :param path_p: path to the vehicle dataset, aka positive example
        :param path_n: path to the non-vehicle data set aka, negative samples
        :return:
        """
        # get the folder names
        p_imgs = glob(path_p+"/**/*.png", recursive=True)
        n_imgs = glob(path_n+"/**/*.png", recursive=True)
        if n_imgs == [] or p_imgs == []:
            raise "No train images found!"
        print("Extracting Positive image samples")
        for img in tqdm(p_imgs):
            self.feat_p.append(self.__extract(mpimg.imread(img)))

        print("Extracting Negative image samples")
        for img in tqdm(n_imgs):
            self.feat_n.append(self.__extract(mpimg.imread(img)))

        self.label_p = np.ones(len(self.feat_p))
        self.label_n = np.ones(len(self.feat_n))


    def __train(self, X,Y ):
        """
        Trains the classifier
        :param X: The input features
        :param Y: The label indicating if the object is present in the image or not
        :return:
        """
        scaled_X = X
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=10)
        self.classifier.fit(X_train,y_train)
        print('Test Accuracy of SVC = ', self.classifier.score(X_test, y_test))

    def predict(self, X):
        return self.classifier.predict(X)

    def __heat_map(self):
        pass

    def classify(self, img):
        """
        The complete classify pipeline, extracts, slidewindow, predict, heatmap
        Boxing
        :param img:
        :return: Returns the image with the boxes applied and and car object

        """
        if self.trained == True:
            return img # TODO: implement
        else:
            # Train
            self.__extract_images()

            self.trained = True
