#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
def __draw_rect(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        plt.imshow(imcopy)
        plt.show()
    # Return the image copy with boxes drawn
    return imcopy
def draw(img, rect):
    pass




def gen_windows(img, x_s_t=[None, None], y_s_t=[None,None], xy_w=(32,32), xy_overlap=(0.4,0.4),sweep=7):
    if None in x_s_t or None in y_s_t:
        x_s_t = [0, img.shape[1]]
        y_s_t = [0, img.shape[0]]

    x_range = x_s_t[1] - x_s_t[0]
    y_range = y_s_t[1] - y_s_t[0]

    #define window sizes
    w_sizes = np.linspace(32,140,7, dtype="int32") #array([  32.,   48.,   64.,   80.,   96.,  112.,  128.]) 
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


if __name__ == '__main__':
    img = plt.imread("sample.jpg")
    w = gen_windows(img, [80, 1200], [375, 700])
    f1,f2= w[0]
    print("-"*20)
    sec = img[f1[0]:f2[0],f1[1]:f2[1]]
    plt.imshow(sec)
    plt.show()
    print("-"*20)
    img_b =  __draw_rect(img,w)
    plt.imshow(img_b)
    plt.show()
    #print("-"*50)
    #print(w)
    #print("-"*50)

