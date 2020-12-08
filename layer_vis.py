#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:47:49 2020

@author: marla

Use to visualize the intermediate layers of vgg16 model
"""


import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras import layers, Model

import matplotlib.pyplot as plt
import math

import numpy as np



def setVisLayers():
    '''
    Function sets the layers that will be visualized; layer names are from 
    pretrained VGG16 model

    Returns
    -------
    plot_layers : list
        A list of layer names.

    '''
    #early layers are better edges
    #plot_layers = ['block2_conv2']
    plot_layers=['block1_conv1',
                 'block1_conv2',
                 'block1_pool', 
                 'block2_conv1',
                 'block3_conv2']
    
    return plot_layers
        
        
def setUpMiniModel(plot_layers, transfer_model):
    '''
    Set up a new model that takes the transfer model inputs and outputs feature
    maps for each of the selected layers

    Parameters
    ----------
    plot_layers : list
        A list of the names of the layers that will be output.
    transfer_model : VGG16 Model
        The pretrained VGG16 model.

    Returns
    -------
    model : keras model
        A keras model with layers specified in plot_layers

    '''
    layerNames = plot_layers
    #a list of outputs
    outputs = [transfer_model.get_layer(name).output for name in layerNames]

    model = tf.keras.Model([transfer_model.input], outputs)
    
    return model 

def displayImages(images, imgTitle, numImages):
	# plot images
    rowSize = math.floor(math.sqrt(numImages))
    colSize = rowSize
    totalImages = rowSize*colSize
    for i in range(totalImages):
        # define subplot
        plt.subplot(rowSize,colSize, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(images[0, :, :, i], cmap='gray_r')
    plt.suptitle(imgTitle)
	
    #show
    plt.show()
    plt.close()
    
def printLayerDetails(model):
    '''
    Function to get the name and output shape for each named layer in the model 

    Parameters
    ----------
    model : keras model
        The pretrained VGG16 model.

    Returns
    -------
    None.

    '''
    for i in range(len(model.layers)):
        layer = model.layers[i]
        print(layer.name, '  ', layer.output.shape)


def read_image(filePath):
    '''
    Read in an image from a file

    Parameters
    ----------
    filePath : string
        The path to the file.

    Returns
    -------
    myImage : np array
        The image as a np array.

    '''
    myImage = load_img(filePath, target_size=(224,224))
    myImage = img_to_array(myImage)
    myImage = myImage.reshape((1, myImage.shape[0], myImage.shape[1], myImage.shape[2]))
    myImage = preprocess_input(myImage)
    
    return myImage
    

def load_pretrained_model():    
    model=VGG16()
    model.summary()
    return model

def visualize_feature_maps(model, myImage, numImages):
    '''
    Function to get the feature map for each of the selected layers 

    Parameters
    ----------
    model : keras model
        The VGG16 model.
    myImage : np array
        The image to visualize.
    numImages : int
        The number of feature maps to display; the display will by nxn, so the
        number of images shown may be fewer than specified

    Returns
    -------
    None.

    '''
    plot_layers = setVisLayers()
    layer_mdl = setUpMiniModel(plot_layers, model)
    
    f_maps = layer_mdl.predict(myImage)
    layer_count = 0
    
    for mp in f_maps:
        print("MAP shape ", mp.shape)
        displayImages(mp, plot_layers[layer_count], numImages)
        layer_count+=1
    
            
        
        
def get_model_prediction(model, myImage):
    '''
    Get the VGG16 class prediction on the image

    Parameters
    ----------
    model : keras model
        VGG16 model.
    myImage : np array
        The image to predict.

    Returns
    -------
    label : tuple
        (id, prediction string, probability)

    '''
    yhat = model.predict(myImage)
    label=decode_predictions(yhat)
    label=label[0][0]
    
    return label

#code to run example
model = load_pretrained_model()
#printLayerDetails(model)

#add the filepath to the image on the line below
path_to_image = ""
myImage = read_image(path_to_image)
numImages = 16
visualize_feature_maps(model, myImage, numImages)
label=get_model_prediction(model, myImage)

print('Prediction: ', label[1])




                               


    
    
