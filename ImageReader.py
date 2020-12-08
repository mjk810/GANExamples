#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:17:57 2020

@author: marla

ImageReader class is used to read and format images from file location for use in CNN

"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import numpy as np

class ImageReader:
    def __init__(self, image_path, image_height, image_width, num_channels):
        self.filePath = image_path
        self.image_list = []
        self.height = image_height
        self.width = image_width
        self.channels=num_channels
        
    def get_image_list(self):
        file_names = os.listdir(self.filePath)
        for file in file_names:
           
            img = mpimg.imread(self.filePath + '/' + file)
            img = self.resizeImage(img)
            #img = self.centerCrop(img)
            #assert img.shape==(128, 128, 3)
            img = self.scaleImage(img)
            
            if img.shape == (self.height, self.width, self.channels):
                self.image_list.append(img)

        
    def resizeImage(self, img):
        #convert to PIL Image to use resize method
        img=Image.fromarray(np.uint8(img))
        wd = self.width
        ht = self.height
        #ch=self.num_channels
        img = img.resize((wd, ht))
        #back to array
        img=np.array(img)
        return img
    
    def centerCrop(self, img):
        '''
        Center crop the images to the desired size

        Parameters
        ----------
        img : nparray
            array of the image.

        Returns
        -------
        cropped image.

        '''
        shp = img.shape
        
        width = img.shape[1]
        height = img.shape[0]
        height_gap = (height - self.height) //2    
        width_gap = (width - self.width)//2
        return img[height_gap:height_gap + self.height, width_gap:width_gap + self.width]
        
    
    def scaleImage(self, img):
	#scale images to range -1 to 1
        img = ((img-127.5)/127.5).astype('float32')
        return img

    
    def formatImages(self):
        pass
    
    def printSample(self):
        
        plt.figure(1)
    
        for i in range(1,26):
            plt.subplot(5,5,i)
            #scale value to 0-1 for plotting
            scaled_img = (self.image_list[i] + 1)/2.0
            plt.imshow(scaled_img)
           
            plt.axis('off')
        
        plt.show()
        
    def getImages(self):
        '''

        Returns
        -------
        TYPE
            NP Array of processed images.

        '''
        return np.array(self.image_list)
 
'''
filePath = '/home/marla/Documents/ganData/collie'
ir = ImageReader(filePath, 64, 64, 3)
ir.get_image_list()
ir.printSample()
im=ir.getImages()
print(im.shape)
#print(np.array(im).shape)

'''
