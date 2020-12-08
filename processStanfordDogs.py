#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:24:50 2020

@author: marla

Use to process Stanford Dogs dataset from file; 
Reads the images using ImageReader, resizes them, and pickles them
"""
import os
import pickle
import numpy as np
from ImageReader import ImageReader



def processStanfordDog():
    filePath = '/home/marla/Documents/gitHub/ComputerVisionProjects/ganData/stanfordDogs/'
    dir_list = os.listdir(filePath)
    all_images = np.empty((0, imageHeight, imageWidth, 3), dtype="float32")
    for d in dir_list:
        fp = filePath + d
        print(fp)
        
        ir = ImageReader(fp, imageHeight,imageWidth, 3)
        ir.get_image_list()
        ir.printSample()
        train_images = ir.getImages()
        
        all_images = np.append(all_images, train_images, axis = 0)
    outfile = '/home/marla/Documents/gitHub/ComputerVisionProjects/ganData/stanfordDogsResize32.sav'
    print("Saved!")
    pickle.dump(all_images,open(outfile, "wb"))
 
    
imageHeight = 32
imageWidth = 32

processStanfordDog()
