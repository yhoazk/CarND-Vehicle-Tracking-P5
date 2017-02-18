#!/usr/bin/env python

from image_processing import *
from svm_classifier import *
import matplotlib.pyplot as plt



if "__main__" == __name__:
    # create the instances of the image processing class and SVM Classifier
    img_proc = Image_Processing()
    #clf = SVM_Classifier()
    #line_lane = lane_detector()

    img_proc.read_video("./project_video.mp4", "png")
    img = img_proc.get_frame(t=3)
    plt.imshow(img)
    plt.show()
    # process the video, take the output of both and add the resutl


