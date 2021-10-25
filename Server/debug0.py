#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:37:41 2021

@author: xyzhou
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM,Flatten, Activation,TimeDistributed, RepeatVector
from tensorflow.keras import regularizers
#import keras
#import keras.backend as K
#from keras.models import load_model
#from keras.models import Model, Sequential
#from keras.layers import LSTM,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
#from keras import regularizers
#from keras.layers import Dropout

dat_reg = np.load('./dat_regression.npz')
X_train = dat_reg['seq_array']
X_test  = dat_reg['seq_array_test_last']
y_Train = dat_reg['label_array']
y_Test  = dat_reg['label_array_test_last']


#idxGroup = [ [0,1,2,12,19], [3,7],[10,15,20,21],[4,8,17],[5,9,13,14],[6,18,22,23],[11,16] ]
idxGroup = [ [0,1,19], [10,15],[4,8,17],[5,9,13,14],[6,22,23],[11,16] ]
# Global, Fan, Splitter, HPC+Fuel, Burner+HPT+LPT, CoreNozzle
grpGlobal = idxGroup[0] 


#grp1 =  grpGlobal+ idxGroup[1] + idxGroup[2]
#grp2 =  grpGlobal+ idxGroup[3]
#grp3 =  grpGlobal+ idxGroup[4] + idxGroup[5]
grp1 =  idxGroup[1] + idxGroup[2]
grp2 =  idxGroup[3]
grp3 =  idxGroup[4] + idxGroup[5]

#X_train1 = X_train[:,:,grp]
X_test1 = X_test[:,:,grp1]
X_test2 = X_test[:,:,grp2]
X_test3 = X_test[:,:,grp3]

pred1 = load_model('pred1_64.h5')
encoder1 = load_model('enc1_64.h5')
latmodel1 = load_model('lat1_64.h5')

pred2 = load_model('pred2_64.h5')
encoder2 = load_model('enc2_64.h5')
latmodel2 = load_model('lat2_64.h5')

pred3 = load_model('pred3_64.h5')
encoder3 = load_model('enc3_64.h5')
latmodel3 = load_model('lat3_64.h5')

#%%
res1 = pred1.predict(X_test1) - y_Test

res2 = latmodel1.predict(  encoder1.predict(X_test1) ) - y_Test

res3 = pred2.predict(X_test2) - y_Test

res4 = latmodel2.predict(  encoder2.predict(X_test2) ) - y_Test

res5 = pred3.predict(X_test3) - y_Test

res6 = latmodel3.predict(  encoder3.predict(X_test3) ) - y_Test

print(np.mean(np.abs(res1)))
print(np.mean(np.abs(res2)))
print(np.mean(np.abs(res3)))
print(np.mean(np.abs(res4)))
print(np.mean(np.abs(res5)))
print(np.mean(np.abs(res6)))
 #%%
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter1 = tf.lite.Interpreter('enc2_64.tflite')
interpreter1.allocate_tensors()

# Get input and output tensors.
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()


# Load TFLite model and allocate tensors.
interpreter2 = tf.lite.Interpreter('lat2_64.tflite')
interpreter2.allocate_tensors()

# Get input and output tensors.
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()


# Test the TensorFlow Lite model on random input data.
input_shape = input_details1[0]['shape']
inputs, outputs = [], []
for i in range(93):
  input_data = np.array(X_test2[i].reshape(1,50,4))
  interpreter1.set_tensor(input_details1[0]['index'], input_data)

  interpreter1.invoke()
  tflite_enc = interpreter1.get_tensor(output_details1[0]['index'])
  
  interpreter2.set_tensor(input_details2[0]['index'], tflite_enc)
  interpreter2.invoke()
  tflite_pred = interpreter2.get_tensor(output_details2[0]['index'])  

  # Test the TensorFlow model on random input data.  
  #output_data = np.array(tf_results)
  
  inputs.append(input_data[0][0])
  outputs.append(tflite_pred[0][0]) 