#!/usr/bin/env python

from image_processing import *
from svm_classifier import *
from lane_detector import *
import matplotlib.pyplot as plt
import cv2
import random as rnd
import matplotlib.image as mpimg


def plot_imgArr(img_arr, label=None, predict=None, gray=False, n=2):
    f, arr = plt.subplots(n, 2)
    print(img_arr[0].shape)
    for n, subplt in enumerate(arr.reshape(-1)):
        if gray:
            subplt.imshow(img_arr[n], cmap='gray')
        else:
            subplt.imshow(img_arr[n])
        subplt.axis('off')
        if label is not None and predict is None:
            subplt.set_title("st: " + str(label[n]))
        elif label is not None and predict is not None:
            subplt.set_title("st:" + str(label[n]) + "p:" + str(predict[n]))
    plt.show()





def test_classifier():
    p_imgs = glob("../vehicles/**/*.png", recursive=True)
    n_imgs = glob("../non-vehicles/**/*.png", recursive=True)

    imgs = []
    labels =[]
    sample = rnd.sample(p_imgs + n_imgs, 8)

    for p in sample:
        img = mpimg.imread(p)
        imgs.append(img)
        pred = clf.classify(img)
        labels.append(pred)

    plot_imgArr(imgs, labels, n=4)




if "__main__" == __name__:
    # create the instances of the image processing class and SVM Classifier
    img_proc = Image_Processing()
    clf = SVM_Classifier()
    line_lane = lane_detector()

    #img_proc.read_video("./project_video.mp4", "jpg")
    # test on single image
    #img = img_proc.get_frame(t=3)
    #clf.classify(img)
    test_classifier()
    print("done")
    exit()
    # Add both images
    img_proc.process(lambda img: cv2.addWeighted(clf.classify(line_lane.camera.undistort(img)), 0.8, line_lane.process(img), 0.4, 0))
    #plt.imshow(img)
    #plt.show()
    # process the video, take the output of both and add the result


