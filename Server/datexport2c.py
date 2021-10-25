#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:27:18 2021

@author: xyzhou
"""

import numpy as np


dat_reg = np.load('./dat_regression.npz')
X_train0 = dat_reg['seq_array']
X_test0  = dat_reg['seq_array_test_last']
y_Train = dat_reg['label_array']
y_Test  = dat_reg['label_array_test_last']


# pick a large window size of 50 cycles
sequence_length = 10
X_train = X_train0[:,-sequence_length:,:]
X_test = X_test0[:,-sequence_length:,:]




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

grp = grp2

X_test = X_test0[:,:,grp]

x1 = X_test[0]
y1 = y_Test[0]
x1c = np.ascontiguousarray(x1, dtype=np.float32)
x1c = np.array2string(x1c.flatten(), precision=6, separator=',',suppress_small=True)

x2 = X_test[6]
y2 = y_Test[6]
x2c = np.ascontiguousarray(x2, dtype=np.float32)
x2c = np.array2string(x2c.flatten(), precision=6, separator=',',suppress_small=True)

x3 = X_test[13]
y3 = y_Test[13]
x3c = np.ascontiguousarray(x3, dtype=np.float32)
x3c = np.array2string(x3c.flatten(), precision=6, separator=',',suppress_small=True)

x4 = X_test[29]
y4 = y_Test[29]
x4c = np.ascontiguousarray(x4, dtype=np.float32)
x4c = np.array2string(x4c.flatten(), precision=6, separator=',',suppress_small=True)
#%%
import re
import hexdump
import sys
def port(array, variable_name='model_data', pretty_print=False):
    array = np.ascontiguousarray(array, dtype=np.float32)
    bytes = hexdump.dump(array).split(',')
    c_array = ', '.join(['0x%02x' % int(byte, 16) for byte in bytes])
    c = 'const unsigned char %s[] DATA_ALIGN_ATTRIBUTE = {%s};' % (variable_name, c_array)
    if pretty_print:
        c = c.replace('{', '{\n\t').replace('}', '\n}')
        c = re.sub(r'(0x..?, ){12}', lambda x: '%s\n\t' % x.group(0), c)
    c += '\nconst int %s_len = %d;' % (variable_name, len(bytes))

    return c
#x1c = port(x1,variable_name='x1', pretty_print=True)


import ctypes

c_float_p = ctypes.POINTER(ctypes.c_float)
data = x1
data = data.astype(np.float32)
data_p = data.ctypes.data_as(c_float_p)


#%%

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

grp = grp3

xTr = X_train[0]
for i in range(14):
    xa = X_train[sequence_length*(i+1)]
    xTr = np.vstack([xTr,xa])

    

# 1st train sample len 142
#xa = X_train0[41]
#xb = X_train0[91]
#xc = X_train0[141]
#xtr1 =  np.vstack([xa,xb,xc])
xtr1 = xTr[:142]
xtr1grp = xtr1[:,grp]
#xtr1grpc = np.ascontiguousarray(xtr1grp, dtype=np.float32)
xtr1grpc = np.array2string(xtr1grp.flatten(), precision=6, separator=',',suppress_small=True)