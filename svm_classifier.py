#!/usr/bin/env python

import cv2
import  matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import svm
from glob import glob
import matplotlib.image as mpimg
from tqdm import tqdm
import time
import pickle


class SVM_Classifier():
    def __init__(self, type='LinearSVC', *clf_params):
        # show debug info
        print(clf_params)
        self.debug = False
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hist_bins = 16
        self.bin_range = (0,256)
        # array to store the windows
        self.windows = []
        #current window
        self.curr_wind = 0
        # stores hog features
        self.hog_array = None
        # defines the search area
        self.image_window = ((70,450),(1210,670))
        self.model_pickle_path = "./clf.p"
        try:
            with open(self.model_pickle_path, "rb") as pick_clf:
                self.classifier = pickle.load(pick_clf)

            with open("scaler.p", "rb") as pick_sclr:
                self.scaler = pickle.load(pick_sclr)
                self.trained = True
        except:
            self.classifier = LinearSVC(C=0.0001, max_iter=5000)
            #self.classifier = svm.SVC( C=1.0, kernel='rbf', max_iter=-1)
            self.trained = False
            self.scaler = None

        # Target resize for each window
        self.tgt_resize = (32,32)
        #type of image to train

        #self positive image dataset
        self.feat_p = []
        self.feat_n = []
        self.label_p = []
        self.label_n = []
        #self.dataset = {'feat_p':self.feat_p, 'feat_n':self.feat_n, 'lbl_p': self.label_p, 'lbl_n': self.label_n}}
        self.dataset = {}
        self.labels = None
        self.features = None

    #def __calc_windows(self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):


    def __get_windows(self, img, x_s_t=[None, None], y_s_t=[None,None], xy_w=(32,32), xy_overlap=(0.4,0.7),sweep=7):
        if None in x_s_t:
            x_s_t = [0, img.shape[1]]
        if None in y_s_t:
            y_s_t = [0, img.shape[0]]

        x_range = x_s_t[1] - x_s_t[0]
        y_range = y_s_t[1] - y_s_t[0]

        #define window sizes
        w_sizes = np.linspace(64,230,5, dtype="int32") #array([  32.,   48.,   64.,   80.,   96.,  112.,  128.])
        #  given the overlap find the y start points
        y_start_points = [y_s_t[0]]
        y_start_points.extend([y_s_t[0]+(1.0-xy_overlap[1])*x for x in w_sizes])
        print(y_start_points)
        windows=[]
        for start_pt, w_size in zip(y_start_points,w_sizes):
            # calculate the number of windows in the first row, and iterate to create them
            # find increment per window
            w_inc = int(w_size * (1.0-xy_overlap[0]))
            wind_per_row = int(x_range/w_inc)
            for x_start in range(x_s_t[0],x_s_t[1],w_inc):
                w = ((int(x_start),int(start_pt)),(int(x_start+w_size), int(start_pt+w_size)))
                if w[1][0] < x_s_t[1]:
                    print(w)
                    windows.append(w)
        return windows


    def __bin_spatial(self, img, color_space='RGB'):
        # Convert image to new color space (if specified)
        img_copy = np.copy(img)
        # supposing we already have the image in RGB
        if color_space == 'HSV':
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2LUV)

        # Use cv2.resize().ravel() to create the feature vector

        features = img_copy.ravel()
        # Return the feature vector
        return features

    def __color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        #
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        #hist_features = channel2_hist[0]
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def __hog(self, img, orient=9, ppc=8, cpb=2, vis=False, f_vect=False):
        hog_img = None
        if vis:
            self.hog_array, hog_img = hog(img,orient,pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),visualise=vis,feature_vector=f_vect)
        else:
            self.hog_array = hog(img,orient,pixels_per_cell=(ppc,ppc),cells_per_block=(cpb,cpb),visualise=vis,feature_vector=f_vect)
        return self.hog_array, hog_img

    def __draw_rect_t(self,img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def __draw_rect(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        # Iterate through the bounding boxes
        imcopy = np.copy(img)
        for n, bbox in enumerate(bboxes):
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        #    cv2.imshow("_", imcopy)
            cv2.imshow("_", imcopy)
            section = imcopy[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            print(
                str(section.shape) + "::" + str(bbox[0][0]) + ":" + str(bbox[1][0]) + "," + str(bbox[0][1]) + ":" + str(
                    bbox[1][1]))
            plt.imsave("sample_" + str(n) + ".png", section)
#            cv2.waitKey(200)
        # Return the image copy with boxes drawn
        #cv2.destroyAllWindows()
        return imcopy


    def __extract(self, img):
        """
        Extracts the features of the image
        :param img: The image from where the features will be extracted, its expected to have the 32x32 image size
        :return: A group of features
        """
        # resize the image to tgt_sze
        # convert to gray for hog
        img = cv2.resize(img, self.tgt_resize)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        feat, img_hog = self.__hog(gray,  orient=9, ppc=8, cpb=3, vis=True, f_vect=False)
        # XXX: The features have a shape 7x7x2x9
        bin_feat = self.__bin_spatial(img)
        hist_feat = self.__color_hist(img) #, nbins=self.hist_bins, bins_range=self.bin_range )
        #self.debug = True
        if self.debug:
            plt.imshow(img_hog)
            plt.show()
        return np.concatenate((bin_feat,  feat.ravel(), hist_feat))
        #return feat.ravel()

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
        self.label_n = np.zeros(len(self.feat_n))

        self.labels = np.hstack((self.label_p, self.label_n))
        ftrs = np.float64(np.vstack((self.feat_p, self.feat_n)))
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(ftrs)
        # save the scaler
        with open("scaler.p", "wb") as scaler_p:
            pickle.dump(self.scaler, scaler_p)
            print("Fitted scaler saved")

        # the data will be shuffled when trainig

    def __train(self, X,Y ):
        """
        Trains the classifier
        :param X: The input features
        :param Y: The label indicating if the object is present in the image or not
        :return:
        """
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
        t = time.time()
        self.classifier.fit(X_train,y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        score = self.classifier.score(X_test, y_test)
        if score > 0.9:
            pickle.dump(self.classifier, open(self.model_pickle_path, "wb"))
            print("Trained classifier saved")
        print('Test Accuracy of SVC = ', self.classifier.score(X_test, y_test))

    def predict(self, X):
        """
        Returns 1 if the prediction is a car 0 otherwise
        :param X:
        :return:
        """
        extracted = self.__extract(X)
        feat = self.scaler.transform(extracted)
        pred =  self.classifier.predict(feat)
        print(pred)
        return pred

    def __heat_map(self):
        pass

    def classify(self, img):
        """
        The complete classify pipeline, extracts, slidewindow, predict, heatmap
        Boxing
        :param img:
        :return: Returns the image with the boxes applied and and car object
        """
        rects = []
        if self.trained != True:
            # Train
            self.__extract_images()
            self.__train(self.features, self.labels)
            self.trained = True

        #calculate the search windows
        self.windows = self.__get_windows(img, y_s_t= [375,670])
        # get the features from the complete image
        glb_feat = None
        for window in self.windows:
            # get the features for the region
            #feat = self._
            # does the region contains a car?
            region = img[window[0][1]:window[1][1],window[0][0]:window[1][0]]
            print("Region:"+ str(region.shape))
            pred = self.predict(region)
            # add the result to the heat map array
            if int(pred[0]) == 1:
                cv2.imshow(str(int(pred[0])),region)
                cv2.waitKey(200)
                rects.append(window)
        cv2.destroyAllWindows()
        print("Windows found: "+ str(len(rects)))
        img_draw = self.__draw_rect(img, rects)
        del rects
        plt.imshow(img_draw)
        plt.show()
        time.sleep(.1)
        #calculate the boxes for each vehicle
        #draw the boxes in the image

        return self.predict(img)

