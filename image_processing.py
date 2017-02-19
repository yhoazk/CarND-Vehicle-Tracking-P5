#!/usr/bin/env python

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import matplotlib.image as npimg


class Image_Processing:
    def __init__(self):
        self.clip = None
        self.proc_clip = None
        self.frame = 0

    def  read_video(self, vid_name, output_fmt="jpg"):
        self.clip = VideoFileClip(vid_name)

    def get_frame(self, t=0, tmp_storage="./test_images/"):
        """
        :param t: Gets a frame at time t and returns it, and as a side effect saves it in test_images
        :return: Image
        """
        if self.clip != None:
            print(tmp_storage+"test_image_"+str(t)+str(self.frame)+".jpg")
            self.clip.save_frame(tmp_storage+"test_image_"+str(t)+str(self.frame)+".jpg")
            img = npimg.imread(tmp_storage+"test_image_"+str(t)+str(self.frame)+".jpg")
            self.frame += 1
        else:
            raise "Call first the read_video passing the name as argin"
        return img
    def process(self, processing_fnc, output_name="out_video"):
        self.proc_clip = self.clip.fl_image(processing_fnc)
        self.proc_clip.write_videofile("./out_video.mp4", audio=False)


