#!/usr/bin/env python

from image_processing import *
from svm_classifier import *
from lane_detector import *
import matplotlib.pyplot as plt
import cv2
import random as rnd
from glob import glob


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
    sample = rnd.sample(p_imgs + n_imgs, 10)

    for p in sample:
        img = clf.get_normImg(p)
        imgs.append(img)
        pred = clf.predict(img)
        labels.append(pred)

    plot_imgArr(imgs, labels, n=5)


def test_classifier1():
    imgs = glob("./window_gen/*.png")

    for p in imgs:
        img = clf.get_normImg(p)
        pred = clf.predict(img)
        if int(pred[0]) == 1:
            cv2.imshow("_-----------im", img)
            cv2.waitKey(500)
        else:
            cv2.imshow("00000000000000", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    # plot_imgArr(imgs, labels, n=5)



def test_classifier2():
    imgs_f = glob("test_images/*.jpg")
    print(imgs_f)

    for p in imgs_f:
        img = clf.get_normImg(p)
        p_img = clf.classify(img)
        #if int(pred[0]) != 0:
        cv2.imshow("_", p_img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    # plot_imgArr(imgs, labels, n=5)

def test_classifier3():
    imgs_f = glob("test_images/*.png")

    for p in imgs_f:
        img = clf.get_normImg(p)
        img = main_fnc(img)
        plt.imshow(img)
        plt.show()
        # img = np.multiply(255, img)
        # img[0][0]
        # cv2.imshow("_", img)
        # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    # plot_imgArr(imgs, labels, n=5)






def main_fnc(img):
    # img = line_lane.camera.undistort(img)

    # test for rgb of video edit tool

    # im = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # return np.dstack((np.zeros_like(im), np.zeros_like(im), im[:,:,1]))
    # the video function reads the video in frames RGB 0-255
    # print(img[0][0])
    img = cv2.normalize(img, img.copy(), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = clf.classify(img)
    img = np.multiply(255,img)
    # cv2.imshow("IMAGE asdasdasdasd", img)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    return img

if "__main__" == __name__:
    # create the instances of the image processing class and SVM Classifier
    img_proc = Image_Processing()
    clf = SVM_Classifier()
    # line_lane = lane_detector()

    # test_classifier()
    #test_classifier()
    #test_classifier()
    # test_classifier1()
    # test_classifier2()
    # test_classifier3()
    img_proc.read_video("project_video.mp4", "jpg")
    img_proc.process(main_fnc)
    exit()
    # img_proc.get_frame(t=20)
    # test on single image
    # img = mpimg.imread("./test_images/test1.jpg")#img_proc.get_frame(t=16)
    # clf.classify(img)
    # img = mpimg.imread("./test_images/test4.jpg")#img_proc.get_frame(t=16)
    # clf.classify(img)
    # img_proc.process(clf.classify)

    print("done")
    # Add both images
    #img_proc.process(lambda img: cv2.addWeighted(clf.classify(line_lane.camera.undistort(img)), 0.8, line_lane.process(img), 0.4, 0))
    #plt.imshow(img)
    #plt.show()
    # process the video, take the output of both and add the result


