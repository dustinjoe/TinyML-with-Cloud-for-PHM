#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:29:01 2021

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

dat_reg = np.load('../dat_regression.npz')
X_train0 = dat_reg['seq_array']
X_test0  = dat_reg['seq_array_test_last']
y_Train = dat_reg['label_array']
y_Test  = dat_reg['label_array_test_last']

# pick a large window size of 50 cycles
sequence_length = 10
X_train = X_train0[:,-sequence_length:,:]
X_test = X_test0[:,-sequence_length:,:]

#%%
cloud_model = load_model('./pred_10step_24ensem.h5')

mcumodelpath = './mlp16_32_16/'
mcu_enc1 = load_model(mcumodelpath+"enc1_32_seq10.h5")
mcu_lat1 = load_model(mcumodelpath+"lat1_32_seq10.h5")
mcu_enc2 = load_model(mcumodelpath+"enc2_32_seq10.h5")
mcu_lat2 = load_model(mcumodelpath+"lat2_32_seq10.h5")
mcu_enc3 = load_model(mcumodelpath+"enc3_32_seq10.h5")
mcu_lat3 = load_model(mcumodelpath+"lat3_32_seq10.h5")
#%%
from model_profiler import model_profiler
from nns.nns import (estimate, printable_dataframe)

Batch_size = 200

units = ['GPU IDs', 'MFLOPs', 'MB', 'Million', 'MB']

print('cloud model:')
#profile_cloud = model_profiler(model=cloud_model, Batch_size=Batch_size, use_units=units)
#print(profile_cloud)
# params = 4 * ((size_of_input + 1) * size_of_output + size_of_output^2)
perf = []
estimate(cloud_model, perf, perf)
printable_dataframe(perf, ignore_phase=False)

print('mcu model1:')
profile_enc1 = model_profiler(mcu_enc1, Batch_size, use_units=units)
print('Enc1: ',profile_enc1)
profile_lat1 = model_profiler(mcu_lat1, Batch_size, use_units=units)
print('Lat1: ',profile_lat1)

print('mcu model2:')
profile_enc2 = model_profiler(mcu_enc2, Batch_size, use_units=units)
print('Enc2: ',profile_enc2)
profile_lat2 = model_profiler(mcu_lat2, Batch_size, use_units=units)
print('Lat2: ',profile_lat2)

print('mcu model3:')
profile_enc3 = model_profiler(mcu_enc3, Batch_size, use_units=units)
print('Enc3: ',profile_enc3)
profile_lat3 = model_profiler(mcu_lat3, Batch_size, use_units=units)
print('Lat3: ',profile_lat3)
#%%
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

def flatDat(X_train1):
    numDat = X_train1.shape[0]
    numStep = X_train1.shape[1]
    numFeat = X_train1.shape[2]
    X_train = X_train1.reshape(numDat,numStep*numFeat)
    return X_train



print(cloud_model.summary())
#idxGroup = [ [0,1,2,12,19], [3,7],[10,15,20,21],[4,8,17],[5,9,13,14],[6,18,22,23],[11,16] ]
idxGroup = [ [0,1,19], [10,15],[4,8,17],[5,9,13,14],[6,22,23],[11,16] ]
# Global, Fan, Splitter, HPC+Fuel, Burner+HPT+LPT, CoreNozzle
grpGlobal = idxGroup[0] 
grp1 =  idxGroup[1] + idxGroup[2]
grp2 =  idxGroup[3]
grp3 =  idxGroup[4] + idxGroup[5]
grpall = grpGlobal+grp1+grp2+grp3
X_train1 = X_train[:,:,grpall]
X_test1 = X_test[:,:,grpall]
print('cloud_model test:')
print(cloud_model.evaluate(X_test1,y_Test))
print('cloud_model train:')

print(cloud_model.evaluate(X_train1,y_Train))


print('Latent predictor (MCU 2nd part) structure:')
print(mcu_lat1.summary())


#%%
from time import time

print('cloud_model train:')
start = time()
cloud_model.evaluate(X_train1,y_Train)
end = time()
timetraininf = end-start


'''
print(mcu_enc1.summary())
print('mcu_model1 test:')
X_test1 = X_test[:,:,grp1]
X_test1 = flatDat(X_test1)
print(mcu_lat1.evaluate( mcu_enc1.predict( X_test1 ),y_Test))
print('mcu_model1 train:')
X_train1 = X_train[:,:,grp1]
X_train1 = flatDat(X_train1)
print(mcu_lat1.evaluate( mcu_enc1.predict( X_train1 ),y_Train))

print(mcu_enc2.summary())
print('mcu_model2 test:')
X_test2 = X_test[:,:,grp2]
X_test2 = flatDat(X_test2)
print(mcu_lat2.evaluate( mcu_enc2.predict( X_test2 ),y_Test))
print('mcu_model2 train:')
X_train2 = X_train[:,:,grp2]
X_train2 = flatDat(X_train2)
print(mcu_lat2.evaluate( mcu_enc2.predict( X_train2 ),y_Train))

print(mcu_enc3.summary())
print('mcu_model3 test:')
X_test3 = X_test[:,:,grp3]
X_test3 = flatDat(X_test3)
print(mcu_lat3.evaluate( mcu_enc3.predict( X_test3 ),y_Test))
print('mcu_model3 train:')
X_train3 = X_train[:,:,grp3]
X_train3 = flatDat(X_train3)
print(mcu_lat3.evaluate( mcu_enc3.predict( X_train3 ),y_Train))
'''
