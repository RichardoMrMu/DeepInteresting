# -*- coding: utf-8 -*-
# @Time    : 2019-11-19 20:28
# @Author  : RichardoMu
# @File    : main.py
# @Software: PyCharm
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import datasets
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_min_log_level"] = '2'
def preprocess(x):
    x = tf.cast(x,dtype=tf.float32)/255.
    # x = tf.reshape(x,(len(x),28,28,1))
    return x
(x_train,_),(x_test,_) = datasets.mnist.load_data()
print(x_train.shape)
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.map(preprocess).batch(128)
# print(train_db.shape)