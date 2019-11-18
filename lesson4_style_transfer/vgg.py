# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 14:35
# @Author  : RichardoMu
# @File    : vgg.py
# @Software: PyCharm
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import layers,optimizers,Sequential,regularizers
import tensorflow as tf
class VGG(keras.Model):
    def __init__(self,num_class,dropout=0.02,weight_decay=0.01,):
        super(VGG, self).__init__()
        # weight_decay = 0.001
        self.num_class = num_class
        self.vgg = Sequential([
            # conv3_64 conv1_1
            layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv3_64 conv1_2
            layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2,2)),

            # conv2_1 conv3_128
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv2_2 conv3_128
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2,2)),

            # conv3-256 conv3-1
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv3-256 con3_2
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv3-256 conv3-3
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv3-256 conv3-4
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3,3)),

            # conv4-1 conv3_512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv4_2 conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv4_3 conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv4_4 conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.MaxPool2D(pool_size=(2,2)),
            # conv5-1  conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv5-2  conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv5-2  conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            # conv5-2  conv3-512
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),

            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),

            layers.Dense(4096,kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout),

            layers.Dense(1000, kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(2, 2))
        ])

    def call(self,inputs,training=None):
        outputs = self.vgg(inputs)
        return outputs