#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:35:33 2021

@author: xyzhou
"""
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

dat_reg = np.load('./dat_regression.npz')
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
def createAuto0(seq_array):
    # Next, we build a deep network. 
    # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
    # Dropout is also applied after each LSTM layer to control overfitting. 
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    nb_features = seq_array.shape[2]
    seq_len = seq_array.shape[1]
    
    BaseDim = 8
    
    model = Sequential()
    
    inputx = Input((seq_len, nb_features,))
    x = LSTM(
         input_shape=(seq_len, nb_features),
         units=12,
         return_sequences=True)(inputx)
    
    x = LSTM(units=6,return_sequences=True)(x)
    x = Dropout(0.2)(x) 
    x = LSTM(units=12,return_sequences=True)(x)
    x = Dropout(0.2)(x) 
    x = (Dense(units=nb_features))(x)
    outx = Activation("linear")(x)    
    
    model = Model(inputx,outx)
    
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    
    
    return model

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(32, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model


# define the autoencoder network model
def predrul_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    Lat = LSTM(16, activation='relu', return_sequences=False,unroll=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    #L1 = Dropout(0.2)(L1) 
    #Lat = LSTM(8, activation='relu', return_sequences=False)(L1)
    
    #L3 = Flatten()(L2)
    L4 = Dense(8, activation='relu')(Lat)
    output = Dense(1,activation='linear')(L4)
    
    model = Model(inputs=inputs, outputs=output)
    
    encoder = Model(inputs, Lat)   
    
 
    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(16,))
    # Retrieve the last layer of the autoencoder model
    L4_dense = model.layers[-2]
    latpred_layer = model.layers[-1]
    # Create the decoder model
    latmodel = Model(encoded_input, latpred_layer(L4_dense(encoded_input)))
   
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model,encoder,latmodel


# define the autoencoder network model
def predrul_conv(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    Lat = LSTM(16, activation='relu', return_sequences=False,unroll=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    #L1 = Dropout(0.2)(L1) 
    #Lat = LSTM(8, activation='relu', return_sequences=False)(L1)
    
    #L3 = Flatten()(L2)
    L4 = Dense(8, activation='relu')(Lat)
    output = Dense(1,activation='linear')(L4)
    
    model = Model(inputs=inputs, outputs=output)
    
    encoder = Model(inputs, Lat)   
    
 
    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(16,))
    # Retrieve the last layer of the autoencoder model
    L4_dense = model.layers[-2]
    latpred_layer = model.layers[-1]
    # Create the decoder model
    latmodel = Model(encoded_input, latpred_layer(L4_dense(encoded_input)))
   
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model,encoder,latmodel

# define the autoencoder network model
def predrul_mlp(X):
    inputs = Input(shape=(X.shape[1], ))
    L1 = Dense(16, activation='relu')(inputs)
    L1 = Dropout(0.2)(L1) 
    #Lat = Dense(16, activation='relu')(L1)
    #L2  = Dropout(0.2)(L2)  
    Lat = Dense(32, activation='relu')(L1)
    #Lat = LSTM(8, activation='relu', return_sequences=False)(L1)
    
    #L3 = Flatten()(L2)
    L4 = Dense(16, activation='relu')(Lat)
    output = Dense(1,activation='relu')(L4)
    
    model = Model(inputs=inputs, outputs=output)
    
    encoder = Model(inputs, Lat)   
    
 
    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(32,))
    # Retrieve the last layer of the autoencoder model
    L4_dense = model.layers[-2]
    latpred_layer = model.layers[-1]
    # Create the decoder model
    latmodel = Model(encoded_input, latpred_layer(L4_dense(encoded_input)))
   
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model,encoder,latmodel


# define the autoencoder network model
def auto_mlp(X):
    inputs = Input(shape=(X.shape[1], ))
    L1 = Dense(64, activation='relu')(inputs)
    L1 = Dropout(0.2)(L1) 
    #Lat = Dense(16, activation='relu')(L1)
    #L2  = Dropout(0.2)(L2)  
    Lat = Dense(32, activation='relu')(L1)
    #Lat = LSTM(8, activation='relu', return_sequences=False)(L1)
    
    #L3 = Flatten()(L2)
    L4 = Dense(64, activation='relu')(Lat)
    output = Dense(X.shape[1],activation='relu')(L4)
    
    model = Model(inputs=inputs, outputs=output)
    
    encoder = Model(inputs, Lat)   
    
 
    # This is our encoded (32-dimensional) input
    encoded_input = Input(shape=(32,))
    # Retrieve the last layer of the autoencoder model
    L4_dense = model.layers[-2]
    latpred_layer = model.layers[-1]
    # Create the decoder model
    latmodel = Model(encoded_input, latpred_layer(L4_dense(encoded_input)))
   
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','mse'])
    return model,encoder,latmodel



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


def trainGrpAuto(X_train,grp,model_path):
    X_train1 = X_train[:,:,grp]
    X_train1 = flatDat(X_train1)
    #autoencoder = createAuto0(X_train1)
    model,encoder,latmodel = auto_mlp(X_train1)
    model.fit(X_train1, X_train1, epochs=100, batch_size=200, validation_split=0.08, verbose=2,
              callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
              )
    #autoencoder.save('autoencoder_drp30.h5')
    #print(model.summary())
    #X_test1 = X_test[:,:,grp]
    #print(model.evaluate(X_test1,X_test1))
    return model,encoder,latmodel


def trainGrpPred(X_train,y_Train,grp,model_path):
    X_train1 = X_train[:,:,grp]
    X_train1 = flatDat(X_train1)
    #autoencoder = createAuto0(X_train1)
    model,encoder,latmodel = predrul_mlp(X_train1)
    model.fit(X_train1, y_Train, epochs=100, batch_size=200, validation_split=0.08, verbose=2,
              callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
              )
    #autoencoder.save('autoencoder_drp30.h5')
    #print(model.summary())
    #X_test1 = X_test[:,:,grp]
    #print(model.evaluate(X_test1,X_test1))
    return model,encoder,latmodel


# define path to save model
model_path = './pred1_16_1_seq10.h5'
pred1,encoder1,latmodel1 = trainGrpPred(X_train,y_Train,grp1,model_path)

# define path to save model
model_path = './pred2_16_1_seq10.h5'
pred2,encoder2,latmodel2 = trainGrpPred(X_train,y_Train,grp2,model_path)

# define path to save model
model_path = './pred3_16_1_seq10.h5'
pred3,encoder3,latmodel3 = trainGrpPred(X_train,y_Train,grp3,model_path)

#%%

print(pred1.summary())
print('pred1 test:')
X_test1 = X_test[:,:,grp1]
X_test1 = flatDat(X_test1)
print(pred1.evaluate(X_test1,y_Test))
print('pred1 train:')
X_train1 = X_train[:,:,grp1]
X_train1 = flatDat(X_train1)
print(pred1.evaluate(X_train1,y_Train))

print(pred2.summary())
print('pred2 test:')
X_test2 = X_test[:,:,grp2]
X_test2 = flatDat(X_test2)
print(pred2.evaluate(X_test2,y_Test))
print('pred1 train:')
X_train2 = X_train[:,:,grp2]
X_train2 = flatDat(X_train2)
print(pred2.evaluate(X_train2,y_Train))

print(pred3.summary())
print('pred2 test:')
X_test3 = X_test[:,:,grp3]
X_test3 = flatDat(X_test3)
print(pred3.evaluate(X_test3,y_Test))
print('pred1 train:')
X_train3 = X_train[:,:,grp3]
X_train3 = flatDat(X_train3)
print(pred3.evaluate(X_train3,y_Train))

#%%

encoder1.save('enc1_24_seq10.h5')
latmodel1.save('lat1_24_seq10.h5')
encoder2.save('enc2_24_seq10.h5')
latmodel2.save('lat2_24_seq10.h5')
encoder3.save('enc3_24_seq10.h5')
latmodel3.save('lat3_24_seq10.h5')



#%%
'''
def trainGrpAuto(X_train,grp,model_path):
    X_train1 = X_train[:,:,grp]
    #autoencoder = createAuto0(X_train1)
    autoencoder = autoencoder_model(X_train1)
    autoencoder.fit(X_train1, X_train1, epochs=100, batch_size=200, validation_split=0.08, verbose=2,
              callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
              )
    #autoencoder.save('autoencoder_drp30.h5')
    print(autoencoder.summary())
    X_test1 = X_test[:,:,grp]
    print(autoencoder.evaluate(X_test1,X_test1))
    return autoencoder

# define path to save model
model_path = './auto1_gt50_32_seq10.h5'
auto1 = trainGrpAuto(X_train,grp1,model_path)

# define path to save model
model_path = './auto2_gt50_32_seq10.h5'
auto2 = trainGrpAuto(X_train,grp2,model_path)

#%% define path to save model
model_path = './auto3_gt50_32_seq10.h5'
auto3 = trainGrpAuto(X_train,grp3,model_path)
'''