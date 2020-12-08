#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:51:59 2020

@author: marla
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, ReLU
import matplotlib.pyplot as plt
import random
from tensorflow.keras.initializers import RandomNormal

from ImageReader import ImageReader
import pickle

import os


    
def define_generator(latent_dim):
    model=Sequential()
    model.add(Dense(512*4*4, input_dim=latent_dim))
    model.add(Reshape((4,4,512)))
    
    
    model.add(Conv2DTranspose(256, kernel_size = 3, strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    
    model.add(Conv2DTranspose(128, kernel_size = 3, strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    model.add(Conv2DTranspose(64, kernel_size = 3, strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    
    #need to output a 64x64x3 image
    model.add(Conv2DTranspose(3, kernel_size = 3, strides=(2,2), padding='same', activation='tanh'))


    print("Generator ")
    model.summary()
    return model


    
def define_discriminator(imgHeight, imgWidth):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding='same', input_shape=(imgHeight, imgWidth, 3)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    opt=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
 #  model.summary()
    return model


def defineGAN(discriminator, generator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print("gan model")
    model.summary()
    return model

def createFakeImages(latent_dim, num_samples, model):
    noise = createNoiseVector(latent_dim, num_samples).astype('float32')
    noise = noise.reshape(num_samples, latent_dim)
    fake_images = model.predict(noise)
    
    return fake_images

def createNoiseVector(latent_dim, num_samples):
    #np.random.normal(mu, sigma)
    noise = np.array([np.random.normal(0,1,latent_dim) for x in range(num_samples)])
    return noise

def getRealImages(num_images, imgs):
    #randomly sample
    n = [random.randint(0,len(imgs)-1) for i in range(num_images)]
    real_images = [imgs[i] for i in n]
    return np.array(real_images)

def generate_positive_labels(y_real, minVal, maxVal):
    
    mod_real=[x*(random.uniform(minVal, maxVal)) for x in y_real]
    
    #mod_real=[x*(minVal+(random.random()*(maxVal-minVal))) for x in y_real]
    return np.array(mod_real)

def generate_negative_labels(y_fake, minVal, maxVal):
    mod_fake=[x+(random.uniform(minVal, maxVal)) for x in y_fake]
    return np.array(mod_fake)

def eval_performance(epoch, discriminator, generator, latent_dim, num_samples, train_images):
    #eval disc on fakes
    xFake = createFakeImages(latent_dim, num_samples, generator)
    yFake = np.zeros((num_samples,1))
    _, fake_acc = discriminator.evaluate(xFake, yFake)
    #eval disc on reals
    xReal = getRealImages(num_samples, train_images)
    yReal = np.ones((num_samples, 1))
    _, real_acc = discriminator.evaluate(xReal, yReal)
    print("Accuracy real: ", real_acc * 100)
    print("Accuracy fake: ", fake_acc * 100)
    #save the model
    model_filename = '/home/marla/Documents/gitHub/ComputerVisionProjects/ganData/largeCollie/Models/generator' + str(epoch)
    generator.save(model_filename)
    
def addNoise(y_values, value_type):
    n=round(y_values.shape[0]*0.05)

    idx = np.random.choice(y_values.shape[0], n, replace=False)
    if value_type=="Real":
        real_vals = generate_positive_labels(np.ones((n,1)), 0.7, 1.0)  
    else:
        real_vals = generate_negative_labels(np.zeros((n,1)),0,0.1)
     
    for i in range(n):
            y_values[idx[i]]=real_vals[i]
            
    return y_values

def xrealcopy(num_images, imgs, img_dim):
    #randomly sample
    n = [random.randint(0,len(imgs)-1) for i in range(num_images)]
    real_images = [imgs[i] for i in n]
    real_images = np.array(real_images)
    real_images.reshape(num_images, img_dim, img_dim, 3)
    return real_images, n

def displayImages(images, imgTitle):
    # scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images
	for i in range(25):
		# define subplot
		plt.subplot(5,5, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(np.around(images[i],2))
	# save plot to file
	#filename = 'generated_plot_e%03d.png' % (epoch+1)
    # save plot to file
	plt.suptitle(imgTitle)
    #show
	plt.show()
	#pyplot.savefig(filename)
	plt.close()
    

  
def train_gan(gan, discriminator, generator, latent_dim, train_images, img_dim):
    num_epochs = 201
    batch_size=128
    half_batch = batch_size//2
    num_batches = int(train_images.shape[0]//batch_size)
    print("batches: ", num_batches)
    
    for i in range(num_epochs):
        for j in range(num_batches):
            #x_temp = train_images.copy()
            #x_temp = x_temp.reshape(train_images.shape[0], img_dim,img_dim, 3)
            #print('thre ', x_temp.shape)
            #displayImages(x_temp, 'test2')
            
            #for j in range(num_batches):
            #create fake images
            xFake = createFakeImages(latent_dim, half_batch, generator)
            #yFake = np.zeros((half_batch,1))
            yFake = generate_negative_labels(np.zeros((half_batch,1)),0,0.1)
            #add noise
            yFake = addNoise(yFake, "Real")
            fake_loss = discriminator.train_on_batch(xFake,yFake)
                     
            #create real images
            xReal = getRealImages(half_batch, train_images)
            #xReal, n = xrealcopy(half_batch, x_temp, img_dim)
            #x_temp = np.delete(x_temp, n,0)
            #x_temp = x_temp.reshape(train_images.shape[0] - (j+1)*batch_size, 32, 32, 3)
            yOnes = np.ones((half_batch, 1))
            yReal = generate_positive_labels(yOnes, 0.8, 1.0)
            yReal = addNoise(yReal, "Fake")
             
            real_loss = discriminator.train_on_batch(xReal,yReal)
            #combine images
            #x=np.vstack((xFake,xReal))
            #y=np.vstack((yFake, yReal))
            #update discriminator weights
            #disc_loss = discriminator.train_on_batch(x,y)
            #update gan
            #create points for gan
             
            #y_gan=np.ones((batch_size,1))
            y_gan = generate_positive_labels(np.ones((batch_size,1)), 0.8, 1.0)
            x_gan = createNoiseVector(latent_dim, batch_size)
            gan_loss = gan.train_on_batch(x_gan,y_gan)
            
            
            
            print("->", str(i), ", ", str(j), "/", str(num_batches), ', f_loss: ', fake_loss[0], ', r_loss: ', real_loss[0], 'g_loss, ', gan_loss[0])
            
            if j==0:
                displayImages(xFake, 'Epoch: '+ str(i) + ' Fake Batch: ' + str(j) )
                displayImages(xReal, 'Epoch: '+ str(i) + ' Real Batch: ' + str(j) )
            
        print("generating ", i)
        generated_images = createFakeImages(100, 50, generator)
        displayImages(generated_images, 'Iteration: '+str(i))
        if i%10 ==0:
            eval_performance(i, discriminator, generator, latent_dim, 100, train_images )
            
        
            
            
    return generator

    
    

#define constants
imageHeight = 64
imageWidth = 64
latent_dim = 100

img_path = '/home/marla/Documents/gitHub/ComputerVisionProjects/ganData/stanfordDogsResizeSmall.sav'
train_images = pickle.load(open(img_path, 'rb'))
train_images = train_images[:10000,:,:,:]
print("NUMBER OF TRAINING IMAGES SHAPE ", train_images.shape)    

#define gan architecture
generator = define_generator(100)
discriminator = define_discriminator(imageHeight, imageWidth)
gan = defineGAN(discriminator, generator)


trained_gen = train_gan(gan, discriminator, generator, latent_dim, train_images, imageHeight)

