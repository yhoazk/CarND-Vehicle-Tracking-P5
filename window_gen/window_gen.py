#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
def __draw_rect(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    # Iterate through the bounding boxes
    img = img[..., ::-1]
    imcopy = np.copy(img)
    for n,bbox in enumerate(bboxes):
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        cv2.imshow("_", imcopy)
        section = imcopy[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
        print(str(section.shape) + "::" + str(bbox[0][0])+":"+ str(bbox[1][0])+","+ str(bbox[0][1])+":"+ str(bbox[1][1]))
        cv2.imwrite("sample_"+str(n)+".png", section)
        cv2.waitKey(20)
    # Return the image copy with boxes drawn
    #cv2.destroyAllWindows()
    return imcopy
def draw(img, rect):
    pass




def gen_windows(img, x_s_t=[None, None], y_s_t=[None,None], xy_w=(32,32), xy_overlap=(0.6,0.5),sweep=7):
    if None in x_s_t:
        x_s_t = [0, img.shape[1]]
    if None in y_s_t:
        y_s_t = [0, img.shape[0]]

    x_range = x_s_t[1] - x_s_t[0]
    y_range = y_s_t[1] - y_s_t[0]

    #define window sizes
    w_sizes = np.linspace(72, 190, 5, dtype="int32") #array([  32.,   48.,   64.,   80.,   96.,  112.,  128.])
    print(w_sizes)
    time.sleep(2)
    #  given the overlap find the y start points
    y_start_points = [y_s_t[0]]
    y_start_points.extend([y_s_t[0]+(1.0-xy_overlap[1])*x for x in w_sizes])
    # print(y_start_points)
    windows=[]
    for start_pt, w_size in zip(y_start_points,w_sizes):
        # calculate the number of windows in the first row, and iterate to create them
        # find increment per window
        w_inc = int(w_size * (1.0-xy_overlap[0]))
        wind_per_row = int(x_range/w_inc)
        for x_start in range(x_s_t[0],x_s_t[1],w_inc):
            w = ((int(x_start),int(start_pt)),(int(x_start+w_size), int(start_pt+w_size)))
            if w[1][0] < x_s_t[1]:
                # print(w)
                windows.append(w)
    return windows


if __name__ == '__main__':
    img = plt.imread("sample_00024.png_")
    w = gen_windows(img, y_s_t= [400, 690])
    f1,f2= w[0]
    print("-"*20)
    print("-"*20)
    img_b =  __draw_rect(img,w)
    plt.imshow(img_b)
    plt.show()
    #print("-"*50)
    #print(w)
    #print("-"*50)

