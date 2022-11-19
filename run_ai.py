#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:03:18 2022

@author: lawray
"""

import monte_carlo.summarize_mc as smc
import ai_optimize.ai_optimize as aio
import tensorflow as tf
import numpy as np
import monte_carlo.monte_carlo as mc

from sklearn.model_selection import train_test_split

# filename= 'TrainingData_hot.p'  # or 'Training Data.p'
filename= 'Training Data.p'  # or 'Training Data.p'
compareFile = 'TrainingData_hot.p'
loadData = True
if loadData:
    mc_out = smc.MonteCarloOutput()
    mc_out.load(filename)
    mc_out.trainingData.scrambleData()
    
    mc_out2 = smc.MonteCarloOutput()
    mc_out2.load(compareFile)
    mc_out2.trainingData.scrambleData()
    
    
    input_shape = (len(mc_out.trainingData.X_s), mc_out.trainingData.X_s[1].shape[0], mc_out.trainingData.X_s[1].shape[1], 1)
    output_length = len(mc_out.trainingData.y[1])
    
    input_shape2 = (len(mc_out2.trainingData.X_s), mc_out2.trainingData.X_s[1].shape[0], mc_out2.trainingData.X_s[1].shape[1], 1)
    output_length2 = len(mc_out2.trainingData.y[1])
    
    mc_out.trainingData.X_s = np.reshape(mc_out.trainingData.X_s,input_shape2)
    mc_out2.trainingData.X_s = np.reshape(mc_out.trainingData.X_s,input_shape2)

x_train, x_test, y_train, y_test = train_test_split(mc_out.trainingData.X_s, np.asarray(mc_out.trainingData.y_s), test_size=0.2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(mc_out2.trainingData.X_s, np.asarray(mc_out2.trainingData.y_s), test_size=0.2)
# y_test /= mc.maxDisplacement
# y_train /= mc.maxDisplacement

discriminator = aio.make_discriminator_model(input_shape, output_length)
# discriminator = aio.make_naive_model(input_shape, output_length)

# choose a loss function from tf.keras.losses?
opt = tf.keras.optimizers.Adam()
# loss_function = tf.keras.losses.MeanSquaredError()
# loss_function = tf.keras.losses.MeanSquaredLogarithmicError() # is there a version msle that uses log(abs(y-y_pred))^2 ???

loss_function = tf.keras.losses.Huber()
loss_function2 = tf.keras.losses.MeanSquaredError()

discriminator.compile(optimizer = opt, loss = loss_function)
discriminator.fit(x=x_train, y=y_train,epochs=5)


y_pred = discriminator.predict(x_test)
y_pred2 = discriminator.predict(x_test2)


y_pred_tmp = y_pred* (abs(y_pred)<5)
y_test_tmp = y_test* (abs(y_test)<5)
y_pred_tmp2 = y_pred2* (abs(y_pred2)<5)
y_test_tmp2 = y_test2* (abs(y_test2)<5)


print('\n\nTest error is: ' + str(float(loss_function(y_pred,y_test))))
print('\n\nTest error for the alternative MC run is: ' + str(float(loss_function(y_pred2,y_test2))))

print('\n\nError in the small-value (|v|<5) sector is: ' + str(float(loss_function(y_pred_tmp,y_test_tmp))))
print('\n\nError in the alternative small-value (|v|<5) sector is: ' + str(float(loss_function(y_pred_tmp2,y_test_tmp2))))

print('\n\nTest MS error is: ' + str(float(loss_function2(y_pred,y_test))))
print('\n\nTest MS error for the alternative MC run is: ' + str(float(loss_function2(y_pred2,y_test2))))

print('\n\nMS error in the small-value (|v|<5) sector is: ' + str(float(loss_function2(y_pred_tmp,y_test_tmp))))
print('\n\nMS error in the alternative small-value (|v|<5) sector is: ' + str(float(loss_function2(y_pred_tmp2,y_test_tmp2))))

#X = tf.random.normal(input_shape)




# decision = discriminator(x)

# print(decision.shape)