#!/usr/bin/env python

from lane import *
from line import *
from img_proc import *

class lane_detector():
    def __init__(self):
        # create an instance for left and right lines
        self.line_l = Line('l')
        self.line_r = Line('r')
        # pass the instances to the lane constructor
        self.lane = Lane(self.line_l, self.line_r)
        self.camera = img_proc()
        self.camera.camera_calibration(
            "/home/porko/workspace/nd_selfDrive/CarND-Advanced-Lane-Lines/camera_cal/calibration*")

    def process(self,img):
        im = self.camera.undistort(img)
        self.line_l.update(im)
        self.line_r.update(im)
        return  self.lane.process_lane(im)
