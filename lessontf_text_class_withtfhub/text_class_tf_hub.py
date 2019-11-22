# -*- coding: utf-8 -*-
# @Author  : RichardoMu
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
# from keras import optimizers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
print("version:",tf.__version__)
print("eager mode :",tf.executing_eagerly())
print("hub version :",hub.__version__)
print("gpu is ","availavle" if tf.config.experimental.list_physical_devices('GPU') else "NOT AVAILABLE")

# download IMBD datasets
# 将训练集按照6：:4比例进行分割，从而最终我们得到15000个
# 训练样本，10000个验证样本和25000个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6,4])
(train_data,validation_data ) ,test_data = tfds.load(
    name = 'imdb_reviews',
    split = (train_validation_split,tfds.Split.TEST),
    as_supervised = True
)
# 探索数据
# 打印前十个样本
train_examples_batch ,train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)
# 构建模型
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding,input_shape=[],
                           dtype=tf.string,trainable=True)
print(hub_layer(train_examples_batch[:3]))
# 现在我们构建完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()
optimize = tf.keras.optimizers.Adam(lr=1e-3)
mdoel.compile(optimizer=optimize,loss=tf.keras.losses.binary_crossentropy(),
              metrics=['accuracy'])
# train model
# use batch = 512 epoch = 20
history = model.fit(train_data.shuffle(10000).batch(512),epochs=20,validation_data=validation_data.batch(512),
                    verbose = 1)
result = model.evaluate(test_data.batch(512),verbose=2)
for name,value in zip(model.metrics_names,result):
    print('%s:%.3f'%(name,value))