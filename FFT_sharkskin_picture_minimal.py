# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:06:55 2018

@author: alex.gansen


"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from skimage import data
import cv2
from scipy.interpolate import interp1d



##########################################
#Scaling
#########################################

#This picture as 1710/10 pixels per mm
scale_px_in_1mm=int(1710/10) #1 px corresponds to ... mm
scale_mm_in_1px=1/scale_px_in_1mm

#########################
# Region
#########################

y_Zoom_middle_min=1500
y_Zoom_middle_max=1600

#####################################################
#Geometry
#####################################################

slit_width=5 # width of the slit die
slit_height=0.5 # height of the slit die
shear_rate=33

#read in picture in greyscale
photo_data=cv2.imread("f-SBR_unfilled_693_g_33_gimp.png",0) 

#Dimension of Figures
fig_dim_x=7
fig_dim_y=5

#show original picture
plt.figure(figsize=(fig_dim_x,fig_dim_y))
plt.title("Original data")
plt.imshow(photo_data,cmap="Greys_r")
plt.show()

########################################################
# Filters
########################################################

# change the contrast (alpha) or brightness (beta) of an image
#https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
photo_data = cv2.convertScaleAbs(photo_data, alpha=2.0, beta=-30)

#to compensate for imhomogenous illumination we use an adaptive gaussian filter
#https://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
photo_data = cv2.adaptiveThreshold(photo_data,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,1001,-10)

#length of whole picture in mm
picture_length_mm=photo_data.shape[1]*scale_mm_in_1px

#Picture with Gaussian filter
plt.figure(figsize=(fig_dim_x,fig_dim_y))
plt.title("Gaussian filter")
plt.imshow(photo_data,cmap='Greys_r')
plt.show()

#Average the pixels in y direction
x_photo_data_middle_mean=photo_data[y_Zoom_middle_min:y_Zoom_middle_max].mean(0)


extrudate_speed_slit=slit_height/6*shear_rate
print('1 mm contains',scale_px_in_1mm,'pixels')
print('1 pixel corresponds to',scale_mm_in_1px,'mm')
print('The picture has a length of',picture_length_mm,'mm')
print('Extrudate speed inside the slit die',extrudate_speed_slit, 'mm/s')
print('The sample segment is extruded within',picture_length_mm/extrudate_speed_slit,'s')


# in my case:
# 1 pixel is recorded every (picture_length_mm/extrudate_speed_slit)/nb_of_pixels=
#=(10.38 mm/2.75 mm/s )/1776 px=3.77 s/1776 px = 0,0021 s.
#So, during 1 second: 1 px/0.0021 s= 476 px are recorded. -> Nyquist freq=476/2=238Hz 
# 
f_nyquist_pixel=(1/((picture_length_mm/(extrudate_speed_slit*photo_data.shape[1]))))/2
print('Number of pixels in x direction=',photo_data.shape[1])
print('picture_length_mm/(extrudate_speed_slit*photo_data.shape[1]))=',picture_length_mm/(extrudate_speed_slit*photo_data.shape[1]))
print('Nyquist frequency from pixels=',f_nyquist_pixel)


t_max=picture_length_mm/extrudate_speed_slit
time=np.linspace(0,t_max,photo_data.shape[1])

#Frequency
freqs=np.fft.fftfreq(photo_data.shape[1])*f_nyquist_pixel*2

##################################################
# FFT
#################################################
# We divide the whole FFT by the respectiv brightness of each region 
FFT_middle_mean=np.fft.fft(x_photo_data_middle_mean)/photo_data[y_Zoom_middle_min:y_Zoom_middle_max].mean()


#########################################################
#Contrast vs time plot
########################################################
plt.figure(figsize=(fig_dim_x,fig_dim_y))
plt.plot(time,x_photo_data_middle_mean,label='Contrast from middle 100 averaged pixels of defect',color="blue",
         linewidth=2.0)
plt.xlabel("Time (s)")
plt.ylabel("Pixel Brightness")
plt.legend()
plt.show()

#########################################################
# FFT plot
########################################################

plt.figure(figsize=(fig_dim_x,fig_dim_y))
plt.plot(freqs,np.abs(FFT_middle_mean),label='|FFT| from middle 100 averaged pixels',color="blue",
         linewidth=2.0)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0,40)
plt.ylim(0,1000)
plt.legend()
plt.show()

#########################################################


