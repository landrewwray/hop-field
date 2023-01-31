#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:55:19 2022

"""

import tensorflow as tf
from tensorflow.keras import layers

# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy as sc
#import os
#import PIL

# import time

numFilters = 32
def make_discriminator_model(input_shape, output_length):

    # numFilters2 = 20
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(numFilters, 3, strides=(1), padding='valid',
                                     input_shape=input_shape[1:]))  # shape --> (60000, 20, 8, 20)

    # model.add(layers.Conv1D(numFilters2, 3, strides=(1), padding='same')) # shape --> (60000, 20, 6, 20)
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))



    model.add(layers.Flatten())  # shape --> (60000, input_shape[1]* (input_shape[3]-2) * 20, 1) ???
    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'relu'))
    model.add(layers.BatchNormalization())  # keep? only useful in the middle?
    # model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'gelu'))
    #model.add(layers.Dropout(0.3))

    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(output_length))  # shape --> (60000, 10 molecules * 5 bonds * 2 distortions) -- effective bond number will sometimes be lower

    return model

def make_naive_model(input_shape, output_length):
    """Naive dense-layer-only model to contrast with 1D convolution and other approaches.

    """

    model = tf.keras.Sequential()
    model.add(layers.Flatten())  # shape --> (60000, input_shape[1]* (input_shape[3]-2) * 20, 1) ???

    model.add(layers.Dense(input_shape[1]* input_shape[2]*5, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(input_shape[1]* input_shape[2]*5, activation='gelu'))
    model.add(layers.Dense(input_shape[1]* input_shape[2], activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(input_shape[1]* input_shape[2], activation='relu'))
    model.add(layers.Dense(output_length))
    
    
    return model
