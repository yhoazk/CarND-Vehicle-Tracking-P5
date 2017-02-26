#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
def __draw_rect(img, bboxes, color=(0, 0, 255), thick=1):
    # Make a copy of the image
    # Iterate through the bounding boxes
    img = img[..., ::-1]
    imcopy = np.copy(img)
    for n,bbox in enumerate(bboxes):
        # Draw a rectangle given bbox coordinates
        # cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        cv2.imshow("_", imcopy)
        section = imcopy[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
        print(str(section.shape) + "::" + str(bbox[0][0])+":"+ str(bbox[1][0])+","+ str(bbox[0][1])+":"+ str(bbox[1][1]))
        cv2.imwrite("sample_"+str(n)+".png", section)
        cv2.waitKey(100)
    # Return the image copy with boxes drawn
    #cv2.destroyAllWindows()
    return imcopy
def draw(img, rect):
    pass


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.2, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def gen_windows(img, x_s_t=[None, None], y_s_t=[None,None], xy_w=(32,32), xy_overlap=(0.5,0.),sweep=7):
    if None in x_s_t:
        x_s_t = [0, img.shape[1]]
    if None in y_s_t:
        y_s_t = [0, img.shape[0]]

    x_range = x_s_t[1] - x_s_t[0]
    y_range = y_s_t[1] - y_s_t[0]

    #define window sizes
    # w_sizes = np.linspace(72, 190, 5, dtype="int32") #array([  32.,   48.,   64.,   80.,   96.,  112.,  128.])
    w_sizes = [72, 72, 164]
    xy_overlap = [0.75, 0.75, 0.75]
    print(w_sizes)
    time.sleep(2)
    #  given the overlap find the y start points
    y_start_points = [y_s_t[0]]
    # y_start_points.extend([y_s_t[0]+(1.0-xy_overlap[1])*x for x in w_sizes])
    y_start_points.extend([y_s_t[0] + ofst for ofst in [20, 20, 20]])
    print(y_start_points)
    windows=[]
    for over, start_pt, w_size in zip(xy_overlap, y_start_points,w_sizes):
        # calculate the number of windows in the first row, and iterate to create them
        # find increment per window
        w_inc = int(w_size * 1.5*(1.0-over))
        print(w_inc)
        wind_per_row = int(x_range/w_inc)
        for x_start in range(x_s_t[0],x_s_t[1],w_inc):
            w = ((int(x_start),int(start_pt)),(int(x_start+w_size*1.5), int(start_pt+w_size*1.2)))
            if w[1][0] < x_s_t[1]:
                # print(w)
                windows.append(w)
    return windows


if __name__ == '__main__':
    img = plt.imread("sample_00043.png_")
    w = gen_windows(img, y_s_t= [390, 700])
    # w = slide_window(img, y_start_stop=[400,690])
    f1,f2= w[0]
    print("-"*20)
    print("-"*20)
    img_b =  __draw_rect(img,w)
    plt.imshow(img_b)
    plt.show()
    #print("-"*50)
    #print(w)
    #print("-"*50)

