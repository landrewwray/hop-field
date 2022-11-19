#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:55:19 2022

@author: lawray
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

def make_GAN_model(input_shape, output_length):

    model.add(layers.Dense(output_length))  # shape --> (60000, 10 molecules * 5 bonds * 2 distortions) -- effective bond number will sometimes be lower
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'relu'))
    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'gelu'))
    model.add(layers.BatchNormalization())  # keep? only useful in the middle?
    model.add(layers.Dense(input_shape[1]* (input_shape[2]-2) * numFilters, activation = 'relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Conv1DTranspose(numFilters, 3, strides=(1), padding='valid',
                                     input_shape=input_shape[1:]))  # shape --> (60000, 20, 8, 20)
# discriminator = keras.Sequential(
#     [
#         keras.layers.InputLayer((28, 28, discriminator_in_channels)),
#         layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.GlobalMaxPooling2D(),
#         layers.Dense(1),
#     ],
#     name="discriminator",
# )

# # Create the generator.
# generator = keras.Sequential(
#     [
#         keras.layers.InputLayer((generator_in_channels,)),
#         # We want to generate 128 + num_classes coefficients to reshape into a
#         # 7x7x(128 + num_classes) map.
#         layers.Dense(7 * 7 * generator_in_channels),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Reshape((7, 7, generator_in_channels)),
#         layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
#     ],
#     name="generator",
# )



def make_naive_model(input_shape, output_length):
    """Notes: momentum for training?  Can/should batch normalization be applied more regularly?

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