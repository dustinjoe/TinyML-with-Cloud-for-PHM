#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:27:18 2020

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

'''

idxSteady =np.where(y_Train>=50)[0]
X_train0 = X_train0[idxSteady,:,:]  # >105 steady, 50-105 degrading,  0-50 critical
X_train = X_train0[:,-sequence_length:,:]
X_test = X_test0[:,-sequence_length:,:]
'''



#%%
from tensorflow.keras.layers import Cropping2D,Concatenate,Reshape
def predrul_ensemble(X,predSub1,predSub2,predSub3):
    inputs = Input(shape=(X.shape[1], X.shape[2])) #(10,17)
    
    inputs_amid = Reshape((X.shape[1], X.shape[2],1))(inputs)
    
    in_global = Cropping2D(cropping=((0,0),(0,14)))(inputs_amid)
    in_global = Reshape((10,3))(in_global)
    
    in_pred1 = Cropping2D(cropping=((0,0),(3,9)))(inputs_amid)
    in_pred1 = Flatten()(in_pred1)
    in_pred2 = Cropping2D(cropping=((0,0),(8,5)))(inputs_amid)
    in_pred2 = Flatten()(in_pred2)    
    in_pred3 = Cropping2D(cropping=((0,0),(12,0)))(inputs_amid)
    in_pred3 = Flatten()(in_pred3)

    predSub1.trainable = False
    predSub2.trainable = False
    predSub3.trainable = False
    out_pred1 = predSub1(in_pred1)    
    out_pred2 = predSub2(in_pred2) 
    out_pred3 = predSub3(in_pred3)     
    
    int_L1 = LSTM(16, activation='relu', return_sequences=True)(in_global)
    int_L2 = LSTM(16, activation='relu', return_sequences=False)(int_L1)
    out_global = Flatten()(int_L2)
    
    interm1 = Concatenate()([out_global,out_pred1,out_pred2,out_pred3])
    #interm1 = 
    interm2 = Dense(16, activation='relu')(interm1)
    interm2 = Dropout(0.2)(interm2)
    #interm3 = Dense(16, activation='relu')(interm2)
    #interm3 = Dropout(0.2)(interm3)    
    output = Dense(1,activation='linear')(interm2)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model    
    




#idxGroup = [ [0,1,2,12,19], [3,7],[10,15,20,21],[4,8,17],[5,9,13,14],[6,18,22,23],[11,16] ]
idxGroup = [ [0,1,19], [10,15],[4,8,17],[5,9,13,14],[6,22,23],[11,16] ]
# Global, Fan, Splitter, HPC+Fuel, Burner+HPT+LPT, CoreNozzle
grpGlobal = idxGroup[0] 


# grp1: physical fan speed, corrected fan speed; total tmp at fan inlet, total pressure in bypass-duct, bypass ratio
# grp2: total tmp HPC outlet, total pres HPC outlet, static pres HPC outlet(Ps30), ratio of fuel to Ps30
# grp3: total tmp LPT outlet, HPT coolant bleed, LPT coolant bleet, physical core speed, corrected fan speed
#grp1 =  grpGlobal+ idxGroup[1] + idxGroup[2]
#grp2 =  grpGlobal+ idxGroup[3]
#grp3 =  grpGlobal+ idxGroup[4] + idxGroup[5]
grp1 =  idxGroup[1] + idxGroup[2]
grp2 =  idxGroup[3]
grp3 =  idxGroup[4] + idxGroup[5]
grpall = grpGlobal+grp1+grp2+grp3

model_path = 'pred_10step_24ensem.h5'
X_train1 = X_train[:,:,grpall]
X_test1 = X_test[:,:,grpall]


#mcu_model1 = load_model("pred1_16_seq10.h5")
#mcu_model2 = load_model("pred2_16_seq10.h5")
#mcu_model3 = load_model("pred3_16_seq10.h5")
mcu_modelpath = './mlp16_32_16/'
mcu_enc1 = load_model(mcu_modelpath+'enc1_32_seq10.h5')
mcu_enc2 = load_model(mcu_modelpath+'enc2_32_seq10.h5')
mcu_enc3 = load_model(mcu_modelpath+'enc3_32_seq10.h5')
model = predrul_ensemble(X_train1,mcu_enc1,mcu_enc2,mcu_enc3)
model.fit(X_train1, y_Train, epochs=100, batch_size=200, validation_split=0.08, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )
    #autoencoder.save('autoencoder_drp30.h5')
    #print(model.summary())
    #X_test1 = X_test[:,:,grp]
print(model.evaluate(X_test1,y_Test))



