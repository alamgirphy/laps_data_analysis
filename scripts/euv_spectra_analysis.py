#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:36:31 2026
@author: alam

"""

#====================================
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class euv_file():
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.avg_profile = None
        self.peak_positions = None
        self.show_peaks = False
        self.nm_axis = None
    
    def load_data(self, flip_x = True):

        with Image.open(self.filepath) as img:
            self.raw_data = np.array(img)
            
        if flip_x:
            self.raw_data = np.fliplr(self.raw_data)  # flip left to right
            
        self.avg_profile = np.mean(self.raw_data, axis = 0)
        self.avg_profile = self.avg_profile - np.min(self.avg_profile) # substract the background
            
        print(f'Data loaded ... Shape: {self.raw_data.shape}, Dtype: {self.raw_data.dtype}')

    
    def select_peaks(self, peak_height=0.5):
        
        if self.avg_profile is None:
            print('avg_profile is None --> Calling plot_2d()...')
            self.plot_2d()
        
        max_val = np.max(self.avg_profile)
        
        peak_height = peak_height*max_val

        peak_indices, properties = find_peaks(self.avg_profile, height=peak_height)
        
        self.peak_positions = peak_indices
        
        print(f'Number of peaks: {len(self.peak_positions)}\nPositions: {self.peak_positions}')
        
        #return self.peak_positions
    
    
    
    def plot_2d(self, show_peaks = False):
        
        self.show_peaks = show_peaks
        
        if self.raw_data is None:
            print('Raw Data is None --> Calling load_data() first')
            self.load_data()
            
        
        if self.nm_axis is not None:
            x_axis = self.nm_axis
            x_label = 'Wavelength (nm)'
        
        else:
            x_axis = np.arange(len(self.avg_profile))
            x_label = 'Number of Pixels'

        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            
        ax[0].imshow(self.raw_data, aspect='auto',
                     extent=[x_axis[0], x_axis[-1], self.raw_data.shape[0], 0])
        
        ax[1].plot(x_axis, self.avg_profile, label='Raw Data')
        ax[1].set_xlabel(x_label)
        ax[1].set_ylabel('Intensity (a.u)')
        ax[1].grid(True, linestyle = '--', alpha=0.4)

        
        if self.show_peaks is True:
            if self.peak_positions is None:
                self.select_peaks()
            
            peak_x = x_axis[self.peak_positions]
            
            ax[1].plot(peak_x, self.avg_profile[self.peak_positions], 'rx', markersize=10,
                       label = 'Selected Peaks')
            
            for p, px in zip(self.peak_positions, peak_x):
                ax[1].axvline(px, color='red', linestyle='--', alpha=0.3)
                
                # annotate peaks
                ax[1].annotate(f'{p}', xy=(px, self.avg_profile[p]), xytext=(0,10), textcoords='offset points',
                               ha='center', fontsize=10, color='red')
        
            ax[1].set_ylim(0,)
            plt.legend()
            plt.show()
    
    
    def change_axis(self, plot=True, show_peaks = False):
        if self.avg_profile is None:
            self.load_data()
        
        #pos_px = [292, 503, 542]
        #pos_nm = [14, 10, 9]
        pos_nm = [8.762, 10.41, 10.79]
        pos_px = [287, 447, 484]
        
        print('Make sure pixel positions and corresponding wavelengths are correct......')
        print(f'Pixel: {pos_px}\nWavelength: {pos_nm}')
        
        coeffs = np.polyfit(pos_px, pos_nm, 2)
        self.calibration = np.poly1d(coeffs)
        
        pixel_axis = np.arange(len(self.avg_profile))
        self.nm_axis = self.calibration(pixel_axis)

        if plot:
            self.plot_2d(show_peaks)
            
            
            
            
























        
        