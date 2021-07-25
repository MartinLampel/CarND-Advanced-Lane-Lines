# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 20:01:21 2021

@author: Martin Lampel
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

from line import Line


ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


class LaneFinder:
    
    def __init__(self, mtx, dist, img_size, src_pts=None, dst_pts=None,
                 nwindows=9, margin=100, minpix=50):
        self.mtx = mtx
        self.dist = dist
        
        self.img_size = img_size
        
        if src_pts is None:
            self.src_pts = np.array([[592, 450], [687,450], 
                                     [1000, 660], [280, 660]], np.float32)
            
        if dst_pts is None:
            self.dst_pts = np.array([[200, 0], [1080, 0], [1080, img_size[1]],
                                     [200, img_size[1]]], np.float32)

            
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.Minv = Minv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)
        
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        
        self.left_line = Line()
        self.right_line = Line()
        
        self.n = 7
        
        
    def get_undistored_image(self, img, outfile=None):
        undistorted =  cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
         
        if outfile is not None:
            cv2.imwrite(outfile, undistorted)
            
        return undistorted
    
    
    def convert(self, img, outfile=None):                        
        
        
        hls_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HLS),( 3,3), 3) 
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        lmag_mask = utils.mag_thresh(hls_img[:, :, 1], mag_thresh=(50,255))
        mag_mask1 = utils.mag_thresh(lab_img[:,:, 0], mag_thresh=(50,255))
        
        ldir_mask = utils.dir_threshold(hls_img[:, :, 1], thresh=(0.5, 1.1))
        sdir_mask = utils.dir_threshold(hls_img[:, :, 2], thresh=(0.5, 1.1))
        s_mask = utils.channel_threshold(img, 2, cv2.COLOR_BGR2HLS, 
                                                  thresh=(100,255))
        l_mask = utils.channel_threshold(img, 1, cv2.COLOR_BGR2HLS, 
                                                  thresh=(60,255))
        
         
        binary = np.zeros_like(s_mask)
        binary[(ldir_mask == 1) & (lmag_mask == 1) | 
               (s_mask == 1 ) & (l_mask == 1) & (sdir_mask == 1 )  |
                (mag_mask1 == 1) ] = 1

        if outfile:
             cv2.imwrite(outfile, binary * 255)
             
        return binary
     
    def warp_image(self, img, show_poly=False, outfile=None, binary=False):
         
        warped = cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)
         
        if show_poly:
            warped = utils.show_poly(warped, self.dst_pts)
            
        if binary:
            warped = warped * 255
         
        if outfile:
            cv2.imwrite(outfile, warped)
             
        return warped
             
         
    def find_lane_pixels(self, binary_warped, outfile=None):
      
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        if outfile:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
            
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height                        
           
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Draw the windows on the visualization image
            if outfile:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]              
               
    
        self.left_line.allx = leftx
        self.left_line.ally = lefty
        
        self.right_line.allx = rightx
        self.right_line.ally = righty
        
        if outfile:
            out_img = self.fit_polynomial(out_img)
            cv2.imwrite(outfile, out_img)           
    
    
    def fit_polynomial(self, out_img=None):
        # Find our lane pixels first
    
        left_fit, resleft, _, _, _ = np.polyfit(self.left_line.ally, self.left_line.allx, 2, full=True)
        right_fit, resright, _, _, _  = np.polyfit(self.right_line.ally, self.right_line.allx, 2, full=True)
    
        resleft /= self.left_line.allx.size
        resright /= self.right_line.allx.size
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.img_size[1]-1, self.img_size[1])
        
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
        
        if out_img is not None:
        
            out_img[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
            out_img[self.right_line.ally, self.right_line.allx] = [0, 0, 255]
            # out_img[ploty, left_fitx] = [0, 255, 255]
            pts =  np.array(np.vstack((left_fitx, ploty)), np.int32).T
            pts = pts.reshape((-1,1,2))
            cv2.polylines(out_img, pts,True,(0,255,255))
            
            pts =  np.array(np.vstack((right_fitx, ploty)), np.int32).T
            pts = pts.reshape((-1,1,2))
            cv2.polylines(out_img, pts,True,(0,255,255))
            
            return out_img
        
        return left_fit, right_fit, left_fitx, right_fitx, resleft, resright
    
    def search_around_poly(self, binary_warped):
        
        #         # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        xfit_left = np.polyval(self.left_line.current_fit, nonzeroy)
        xfit_right = np.polyval(self.right_line.current_fit, nonzeroy)
        
        left_lane_inds = (nonzerox > xfit_left - self.margin) & (nonzerox < xfit_left + self.margin)
        right_lane_inds = (nonzerox > xfit_right - self.margin) & (nonzerox < xfit_right + self.margin)
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        self.left_line.allx = leftx
        self.left_line.ally = lefty
        
        self.right_line.allx = rightx
        self.right_line.ally = righty
    
    def find_lanes(self, img, outfile=None):
        
        undistorted = self.get_undistored_image(img) 
        img = self.convert(undistorted)
        binary_warped = self.warp_image(img)       
        
        if self.left_line.detected:
            self.search_around_poly(binary_warped)
        else:
            self.find_lane_pixels(binary_warped, outfile)
            self.left_line.detected = True
            self.right_line.detected = True          
        
       
        left_fit, right_fit, left_fitx, right_fitx, resleft, resright = self.fit_polynomial()
        # print('left_fit, right_fit)
 
        curverad_f = lambda p, y:  (1 + (2*p[0] * y + p[1])**2)**(3/2) /(2*np.abs(p[0]))
        # left_curverad = curverad_f(left_fit, self.img_size[1]-1)
        # right_curverad = curverad_f(right_fit, self.img_size[1]-1)
      
        
        # left_fit, right_fit = self.model_correction(left_fit, right_fit,
        #                                             resleft, resright, 
        #                                             left_curverad,
        #                                             right_curverad)
        
        self.left_line.current_fit = left_fit
        self.right_line.current_fit = right_fit
        self.left_line.radius_of_curvature = curverad_f(self.left_line.best_fit, 
                                                        self.img_size[1]-1 * ym_per_pix) 
        self.right_line.radius_of_curvature = curverad_f(self.right_line.best_fit, 
                                                        self.img_size[1]-1 * ym_per_pix) 

        
        self.left_line.diffs = np.abs(self.left_line.best_fit - left_fit)
        self.right_line.diffs = np.abs(self.right_line.best_fit - right_fit)
        
        # print(np.average(self.left_line.diffs), np.average(self.right_line.diffs))
        
        if (np.mean(self.left_line.diffs + self.right_line.diffs) > 10
            	and self.left_line.first_fit):
            self.left_line.detected = False
            self.right_line.detected = False 
            
            self.left_line.current_fit = self.left_line.best_fit
            self.right_line.current_fit = self.right_line.best_fit
        else:           
            self.left_line.recent_xfitted.append(left_fitx)
            self.right_line.recent_xfitted.append(right_fitx)        
          
            self.left_line.recent_fitt_coeffs.append(left_fit)
            self.right_line.recent_fitt_coeffs.append(right_fit)
        
        if self.left_line.first_fit:
               
            self.left_line.best_fit = left_fit
            self.right_line.best_fit = right_fit
            self.left_line.first_fit = True
            self.right_line.first_fit = True
            
        
        if len(self.left_line.recent_xfitted) > self.n:           
            
            self.left_line.bestx = np.mean(self.left_line.recent_xfitted, axis=0)
            self.right_line.bestx = np.mean(self.right_line.recent_xfitted, axis=0)  
            
            self.left_line.recent_xfitted.clear()
            self.right_line.recent_xfitted.clear()
            
            self.left_line.recent_xfitted.append(self.left_line.bestx)
            self.right_line.recent_xfitted.append(self.right_line.bestx)            
          
            
        if len(self.left_line.recent_fitt_coeffs) > self.n:
            self.left_line.best_fit = np.mean(self.left_line.recent_fitt_coeffs, axis=0)
            self.right_line.best_fit = np.mean(self.right_line.recent_fitt_coeffs, axis=0)
            
            self.left_line.recent_fitt_coeffs.clear()
            self.right_line.recent_fitt_coeffs.clear()
            
            self.left_line.current_fit = self.left_line.best_fit
            self.right_line.current_fit = self.right_line.best_fit

                
        y = np.linspace(0, self.img_size[1]-1, self.img_size[1])
        out_img = self.draw_lanes(undistorted, np.polyval(self.left_line.current_fit, y), 
                                    np.polyval(self.right_line.current_fit, y))
        
        out_img = self.draw_lane_info(out_img)        
       
        # cv2.putText(out_img, '{:.3f} {:.3f} {:.3f} {:.1f}'.format(
        #      self.left_line.current_fit[0], self.left_line.current_fit[1], self.left_line.current_fit[2],np.std(self.left_line.recent_xfitted[-1])),
        #             (50, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # cv2.putText(out_img, '{:.3f} {:.3f} {:.3f} {:.1f}'.format(
        #      self.right_line.current_fit[0], self.right_line.current_fit[1], self.right_line.current_fit[2],np.std(self.right_line.recent_xfitted[-1])),
        #             (50, 80), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return out_img
        
        
    def draw_lanes(self, undist_img, left_fitx, right_fitx):
        
        ploty = np.linspace(0, self.img_size[1]-1, self.img_size[1] )
         
        warp_zero = np.zeros_like(undist_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(warp_zero, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(warp_zero, self.Minv, (self.img_size[0], self.img_size[1])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)
       
        return result
    
    def draw_lane_info(self, img):
        
        curv = (self.left_line.radius_of_curvature + 
                  self.right_line.radius_of_curvature)/2
        
        
        xl = self.left_line.recent_xfitted[-1][self.img_size[1]-1]        
        xr = self.right_line.recent_xfitted[-1][self.img_size[1]-1]
        color = (255, 255, 255)
        xtextcoord = int(self.img_size[0]/2-150)
        
        infobox = np.zeros_like(img)
        infobox[20:100,xtextcoord-60:xtextcoord+350] = color
        img = cv2.addWeighted(img, 1, infobox, 0.2, 0)
        
        lane_center = (xl + xr) / 2
        car_center = self.img_size[1]/2
        
        offset = np.abs(lane_center - car_center) * xm_per_pix
        offsettxt = 'Car is {} from Center: {:0.2f} m'
        curvetxt = 'Curvature {0:0.3f}m'
        
        
        if lane_center > car_center:
            offsettxt = offsettxt.format('left', offset)
        elif lane_center < car_center:
            offsettxt = offsettxt.format('right', offset)
        else:
            offsettxt = 'Car is in Center'
            xtextcoord = int(self.img_size[0]/2-100)          
        
        if (np.std(self.left_line.recent_xfitted[-1]) < 50 and
            np.std(self.right_line.recent_xfitted[-1]) < 50):
            curvetxt = 'straight line'
                 
       
            
        cv2.putText(img, offsettxt, (xtextcoord, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(img, curvetxt.format(curv),
                    (int(self.img_size[0]/2-100), 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
        return img
        
        
            
    def model_correction(self, left_fit, right_fit, 
                          resleft, resright, left_curverad, right_curverad):
        
        lane_distance = np.abs(np.polyval(left_fit, self.img_size[1] - 1 )-
                               np.polyval(right_fit, self.img_size[1] - 1 ))
        
        
        if (self.left_line.allx.size > self.right_line.allx.size 
            or resleft < resright and 
                self.right_line.allx.size  <= 2* self.left_line.allx.size):
                    
            curverad = 0
            if left_curverad  > right_curverad:
                curverad = left_curverad - lane_distance
            else:
                curverad = left_curverad + lane_distance
                
            right_fit = self.correct_params(left_fit, right_fit, curverad)
        else:
            
            curverad = 0
            if left_curverad  > right_curverad:
                curverad = right_curverad + lane_distance
            else:
                curverad = right_curverad - lane_distance
                
            left_fit = self.correct_params(right_fit, left_fit, curverad)
                    
        
            
        return left_fit, right_fit
                    
    
    def correct_params(self, fit, other_fit, curverad):
        
        ys = self.img_size[1] - 1 
        xs = np.polyval(other_fit, ys)
        
        s = np.sign(np.polyval(np.polyder(fit, 2), ys))
        
        dx_dy = np.polyval(np.polyder(fit, 1), ys)
        
        a = s*(1 + dx_dy**2)**(3/2) / 2 / curverad
        b = dx_dy - 2 * a * ys
        c = xs - a * ys**2 - b * ys
        
        return np.array([a, b, c])
                    
            
        
        
        
        
        