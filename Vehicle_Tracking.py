#!/usr/bin/env python

from image_processing import *
from svm_classifier import *
from lane_detector import *
import matplotlib.pyplot as plt
import cv2


if "__main__" == __name__:
    # create the instances of the image processing class and SVM Classifier
    img_proc = Image_Processing()
    clf = SVM_Classifier()
    line_lane = lane_detector()

    img_proc.read_video("./project_video.mp4", "jpg")
    # test on single image
    img = img_proc.get_frame(t=3)
    clf.classify(img)
    print("done")
    exit()
    # Add both images
    img_proc.process(lambda img: cv2.addWeighted(clf.classify(line_lane.camera.undistort(img)), 0.8, line_lane.process(img), 0.4, 0))
    #plt.imshow(img)
    #plt.show()
    # process the video, take the output of both and add the result


