#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:24:06 2021

@author: xyzhou
"""

import re
import hexdump
import tensorflow as tf

    model = load_model(input_keras_model, compile = False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_model_file, "wb").write(tflite_model)

