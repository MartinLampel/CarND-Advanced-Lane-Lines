# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:43:26 2021

@author: lampe_000
"""

import glob
import matplotlib.image as mpimg
import os
import utils
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from lanefinder import LaneFinder
from moviepy.editor import VideoFileClip


CAL_IMAGE_NAMES = glob.glob('./camera_cal/calibration*.jpg')
TEST_IMGAGE_DIR = 'test_images'
OUT_DIR = 'output_images'

DO_CALIBRATION = False

def pipeline_test(imgdir, outdir, mtx, dist):
   
  
    
    for img_path in os.listdir(imgdir):
        lanefinder = LaneFinder(mtx, dist, (1280, 720))
       
        
        basename = os.path.splitext(img_path)[0]
        
        img = cv2.imread(os.path.join(imgdir, img_path))
        
       
         
        outfile = os.path.join(outdir, '{}_undistored.jpg')

        undistorted = lanefinder.get_undistored_image(img, 
                            outfile=outfile.format(basename))

        
        outfile = os.path.join(outdir, '{}_binary_mask.jpg')

        mask = lanefinder.convert(img, 
                            outfile=outfile.format(basename))
        
        
        outfile = os.path.join(outdir, '{}_warped.jpg')

        warped = lanefinder.warp_image(undistorted, show_poly=True,
                            outfile=outfile.format(basename))
        
        cv2.imwrite(os.path.join(outdir, basename + '_poly.jpg'),
                    utils.show_poly(undistorted, lanefinder.src_pts))
        
        outfile = os.path.join(outdir, '{}_warped_mask.jpg')
        
        
        warped = lanefinder.warp_image(mask, outfile=outfile.format(basename), 
                                       binary=True)
        
        outfile = os.path.join(outdir, '{}_lanes.jpg')
        
        final_img = lanefinder.find_lanes(img, outfile=outfile.format(basename))
        
        cv2.imwrite(os.path.join(outdir, basename + '_final.jpg'), final_img)
        
        
        

class Extractor:
    def __init__(self):
        self.out = './challenge/{}.jpg'
        self.n = 0
    def extract(self, img):
        cv2.imwrite(self.out.format(self.n), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.n += 1
        
        return img
    

if __name__ == '__main__':
    
    cal_images = [mpimg.imread(name) for name in CAL_IMAGE_NAMES]
    showimg = [1, 5]#[1, 5, 10, 17]
    
    if DO_CALIBRATION:
        _, mtx, dist, _, _ = utils.calibrate_camera(cal_images, nx=9, ny=6, 
                                          showimg=showimg)
        
        for s in showimg:
            utils.undistort(cal_images[s], mtx, dist, show=True)
    
        data = {'mtx' : mtx, 'dist':dist}
        with open('camera_cal.pickle','wb') as f:
            pickle.dump(data, f)
    else:
        with open('camera_cal.pickle', 'rb') as f:
            data = pickle.load(f)
        mtx = data['mtx']
        dist = data['dist']
        
    # pipeline_test(TEST_IMGAGE_DIR, OUT_DIR, mtx, dist)
    
    # videopath = 'project_video.mp4'
    # lanefinder = LaneFinder(mtx, dist, (1280, 720))
    # video_clip = VideoFileClip(videopath)
    # out_clip = video_clip.fl_image(lanefinder.find_lanes) #NOTE: this function expects color images!!
    # out_clip.write_videofile('project_video_with_lanes1.mp4', audio=False)
    
    # videopath = 'challenge_video.mp4'
    # extractor = Extractor()
    # lanefinder = LaneFinder(mtx, dist, (1280, 720))
    # video_clip = VideoFileClip(videopath)
    # out_clip = video_clip.fl_image(extractor.extract) #NOTE: this function expects color images!!
    # out_clip.write_videofile('challenge_video_with_lanes.mp4', audio=False)
    
    pipeline_test('challenge', 'challenge_out', mtx, dist)
    
    
    
    
     
   
        
    
    
    


