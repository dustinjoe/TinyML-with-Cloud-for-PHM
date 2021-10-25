import tensorflow as tf
from tensorflow.keras.models import load_model



'''
input_keras_model = 'predictive_maintenance.h5'

tflite_model_file = "predictive_maintenance_converted_model.tflite"
model = load_model(input_keras_model, compile = False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_model_file, "wb").write(tflite_model)
'''




import glob
import pathlib

for filepath in glob.iglob('./*.h5'):
    print(filepath)
    input_keras_model = filepath
    filename = filepath[:-3]  #remove .h5
    tflite_model_file = filename+".tflite"
    model = load_model(input_keras_model, compile = False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path(tflite_model_file)
    tflite_model_file.write_bytes(tflite_model)
    #open(tflite_model_file, "wb").write(tflite_model)

#%%
import re
import hexdump
import sys

def port(model, optimize=False, variable_name='model_data', pretty_print=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if optimize:
        if isinstance(optimize, bool):
            optimizers = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        else:
            optimizers = optimize
        converter.optimizations = optimizers
    tflite_model = converter.convert()
    bytes = hexdump.dump(tflite_model).split(' ')
    c_array = ', '.join(['0x%02x' % int(byte, 16) for byte in bytes])
    c = 'const unsigned char %s[] DATA_ALIGN_ATTRIBUTE = {%s};' % (variable_name, c_array)
    if pretty_print:
        c = c.replace('{', '{\n\t').replace('}', '\n}')
        c = re.sub(r'(0x..?, ){12}', lambda x: '%s\n\t' % x.group(0), c)
    c += '\nconst int %s_len = %d;' % (variable_name, len(bytes))
    preamble = '''
// if having troubles with min/max, uncomment the following
// #undef min    
// #undef max
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif
'''
    return preamble + c

for filepath in glob.iglob('*.h5'):
    print(filepath)
    input_keras_model = filepath
    model = load_model(input_keras_model, compile = False)
    filename = filepath[:-3]  #remove .h5
    ccode = port(model, optimize=False, variable_name=filename, pretty_print=True)
    print(ccode)
    
    myCodeFile = open(filename+'.h','w')
    myCodeFile.write(ccode)
    myCodeFile.close()

