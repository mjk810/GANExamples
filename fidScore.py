#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:54:01 2020

@author: marla

This script calculates the FID score to assess the quality of the GAN model. 
The calculation compares the quality of the generated images with actual images.
"""

import tensorflow as tf
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import pickle
from scipy.linalg import sqrtm
import math



class FID(object):
    def __init__(self, imageHeight = 299, imageWidth = 299):
        self.model = tf.keras.applications.InceptionV3(
                        include_top=False,
                        weights="imagenet",
                        pooling="avg"
                    )
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
    
    #utility function to print model layer names
    def printModel(self):
        for layer in self.model.layers:
            print(layer.name)
    
    def resizeImages(self, imgs):
        #images need to be 299 x 299 x 3 per documentation
        resizedImages = tf.image.resize(imgs, [self.imageHeight, self.imageWidth])
        print("NEW SHAPE ", resizedImages.shape)
        return np.asarray(resizedImages)
    
    def computeFid(self, real_images, fake_images):
        #fid  d^2  = ||mu1-mu2|| ^2 + Tr(C1 + C2 - 2*sqrt(C1*C2)) mean and cov
        #first term is sum squared difference
        
        #get the activations from the model; resultant shape is (n_samples,2048)
        pred_real = self.model.predict(real_images)
        pred_fake = self.model.predict(fake_images)
        
        #calculate the means for each row (example); resultant shape is (2048,)
        mu1 = pred_real.mean(axis=0)
        mu2 = pred_fake.mean(axis=0)
        
        #calculate covariances; rowvar = False indicates that the rows are the observations
        #and cols are variables
        C1 = np.cov(pred_real, rowvar = False)
        C2 = np.cov(pred_fake, rowvar = False)
        
        #calculate the sum squared difference between the means for each observation
        ssd = np.sum((mu1 - mu2) **2.0)
        
        #np sqrt won't return complex numbers unless one of the inputs is complex; returns nan
        #so using scipy matrix square root; square root of - will be an imaginary number
        covmat = (sqrtm(np.matmul(C1, C2)))
        
        #replace complex numbers with real numbers
        covmat = covmat.real
         
        d2 = ssd + np.trace(C1 + C2 - 2.0*covmat)
        print("FID ", d2)
        return d2
    
    def testShape(self):
        im = Image.open("")
        resizedImg = self.resizeImages(np.expand_dims(np.asarray(im), axis=0))
        assert resizedImg.shape == (1, self.imageHeight, self.imageWidth, 3)
        print("Passed")
        
    def testSameImageScore(self,real_images):
        fid = self.computeFid(real_images, real_images)
        print("FID = ", round(fid))
        if abs(round(fid)) == 0:
            print("Test Passed!")
            
    def runTests(self, images):
        #print("Testing the image resize function...")
        #self.testShape()
        print("Testing score for the same images; score should be 0...")
        self.testSameImageScore(images)

#example code
#the number of samples to use in the comparison; should be higher, but compute power is limited
num_samples = 100
score = FID()

#get the images used for training
#img_path is the path to real images; add the filepath on the line below
img_path = ''
train_images = pickle.load(open(img_path, 'rb'))
train_images = train_images[:num_samples,:,:,:]
print("NUMBER OF TRAINING IMAGES SHAPE ", train_images.shape)
scaled_reals = score.resizeImages(train_images)
print("SCALED Image shape: ", scaled_reals.shape)

#use the trained gan to generate fake images and scale them for the inception model; specify filename
#on line below
model_filename = ''
ml = tf.keras.models.load_model(model_filename)
noise = np.array([np.random.normal(0,1,100) for x in range(num_samples)])
gen_ims = ml.predict(noise)
scaled_gen = score.resizeImages(gen_ims)
print("SCALED Generated Image shape: ", scaled_gen.shape)

#run the tests
score.runTests(scaled_reals)

#calculate score
score.computeFid(scaled_reals, scaled_gen)
