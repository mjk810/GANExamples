#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:29:12 2020

@author: marla

Wasserstein GAN
Wasserstein GAN uses an alternative loss function based on earth movers distance; can be used to avoid mode collapse
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import pickle
from tensorflow.keras import backend
from tensorflow.keras.constraints import Constraint



'''
    sample code for setting constraints from https://keras.io/api/layers/constraints/

'''
class WeightClip(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be between +/- ref_val `ref_value`."""

  def __init__(self, ref_value):
    self.ref_value = ref_value

  def __call__(self, w):
    return backend.clip(w, -self.ref_value, self.ref_value)

  def get_config(self):
    return {'ref_value': self.ref_value}

def loadImages():
   (train_images, train_labels), (_, _) = mnist.load_data()
   train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
   train_images = (train_images-127.5)/127.5

   return train_images, train_labels


def displayImages(images, imgTitle):
    plt.figure(1)
    
    for i in range(1,26):
        plt.subplot(5,5,i)
        plt.imshow(images[i].reshape([28,28]))
       
        plt.subplot(5,5,i)
        plt.imshow(images[i].reshape([28,28]))
        plt.axis('off')
    plt.suptitle(imgTitle)
    plt.show()

    
   

def generator(latent_dim):
    model = Sequential()
    nodes = 128*7*7 #6272
    model.add(Dense(nodes, input_dim = latent_dim, activation='relu'))
    model.add(Reshape((7,7,128)))
    #upsample to 14*14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    #upsample to 28*28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    #conv layer
    model.add(Conv2D(1, kernel_size=7, activation='tanh', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    #model.summary()
    return model
    
    
def discriminator():
    '''
    Wasserstein Updates
    -------------------
    linear output layer
    RMSProp optimizer
    
    Returns
    -------
    model : critic model.

    '''
    clip = WeightClip(0.01)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_constraint=clip, input_shape=(28,28,1)))
    
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_constraint = clip, activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(1, activation='linear'))
    opt=tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    model.compile(optimizer=opt, loss=wasserstein_loss, metrics=['accuracy'])
    
    #model.summary()
    return model
   

def gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt=tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
   
    return model

def createFakeImages(latent_dim, num_samples, model):
    noise = createNoiseVector(latent_dim, num_samples)
    fake_images = model.predict(noise)
    return fake_images

def createNoiseVector(latent_dim, num_samples):
    noise = np.array([np.random.normal(0,1,latent_dim) for x in range(num_samples)])
    return noise

def getRealImages(num_images, imgs):
    #randomly sample
    n = [random.randint(0,len(imgs)-1) for i in range(num_images)]
    real_images = [imgs[i] for i in n]
    return np.array(real_images)

def wasserstein_loss(y_true, y_pred):
    '''
    Multiply the label by the mean value

    Parameters
    ----------
    y_true : real y
    y_pred : predicted y

    Returns
    -------
    TYPE
        wasserstein_loss

    '''
    return backend.mean(y_true) * backend.mean(y_pred)


def train(gan, discriminator, generator, latent_dim, train_images):
    num_epochs = 20000
    batch_size=64
    half_batch = batch_size//2
    num_batches = int(train_images.shape[0]//batch_size)
    critic_train_steps=5
    
    for i in range(num_epochs):
        for j in range(critic_train_steps):
            #create fake images
            xFake = createFakeImages(latent_dim, half_batch, generator)
            yFake = np.ones((half_batch,1))
            #create real images
            xReal = getRealImages(half_batch, train_images)
            yReal = np.ones((half_batch,1))*-1
            #combine images
            #x=np.vstack((xFake,xReal))
            #y=np.vstack((yFake, yReal))
            #update discriminator weights
            discriminator.train_on_batch(xReal,yReal)
            discriminator.train_on_batch(xFake,yFake)

        
        #update gan
        #create points for gan
        y_gan = np.ones((batch_size,1))*-1
        x_gan = createNoiseVector(latent_dim, batch_size)
        gan_loss = gan.train_on_batch(x_gan,y_gan)
            
        if i%100==0:
            print("generating ", i)
            generated_images = createFakeImages(100, 100, generator)
            displayImages(generated_images, 'Iteration: '+str(i))
    return generator
            

def main():
    latent_dim = 100
    train_images, train_labels = loadImages()
    displayImages(train_images, 'Initial Train Images')
   
    #test out creating the generator and fake_images; show the images created
    gen = generator(latent_dim)
    #fake_images=createFakeImages(100, 100, gen)
    #displayImages(fake_images)
    
    #printing 25 images at a time; no error handling for fewer than 25
    #show the real images to test
    #real_images = getRealImages(26, train_images)
    #displayImages(real_images)
    
    #test out discriminator
    disc=discriminator()
    #predictions = disc.predict(real_images)
    #print("Predictions ", predictions)
    
    ganMdl = gan(gen, disc)
    trained_gen = train(ganMdl, disc, gen, latent_dim, train_images)
    pickle.dump(trained_gen, open('mnist_gen.sav', 'wb'))
    
main()
