# -*- coding: utf-8 -*-
# @Author  : RichardoMu
import tensorflow as tf
import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# use imdb

# step 1 load imdb
# 只使用最频繁的10000个词汇
total_words = 10000
embedding_len = 100
batch_size = 512
lr = 1e-3
max_len = 256
(train_data,train_label),(test_data,test_label) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# get validation datasets
val_data = train_data[:10000]
train_data = train_data[10000:]
val_label = train_label[:10000]
train_label = train_label[10000:]
# preprocessing padding
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,value=0,padding='post',maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,value=0,padding='post',maxlen=256)
val_data = tf.keras.preprocessing.sequence.pad_sequences(val_data,value=0,padding='post',maxlen=256)
train_db = tf.data.Dataset.from_tensor_slices((train_data,train_label))
# print(train_db,train_db.shape)
train_db = train_db.shuffle(15000).batch(batch_size=batch_size)
val_db = tf.data.Dataset.from_tensor_slices((val_data,val_label))
val_db = val_db.shuffle(25000).batch(batch_size=batch_size)
test_db = tf.data.Dataset.from_tensor_slices((test_data,test_label))
test_db = test_db.batch(batch_size=batch_size)
optimize = tf.keras.optimizers.Adam(lr=lr)
# build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words,embedding_len,input_length=max_len))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(64,activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(32,activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(16,activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=optimize,loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
history = model.fit(train_db,epochs=40,validation_data=val_db)
result = model.evaluate(test_db)
# draw pictures
history_dict = history.history
# accuracy loss val_accuracy val_loss
accuracy = history_dict['accuracy']
loss = history_dict['loss']
val_accuracy = history_dict['val_accuracy']
val_loss = history_dict['val_loss']
epochs = range(1,len(accuracy)+1)
# accuarcy
plt.plot(epochs,accuracy,'b',label='Training accuracy')
plt.plot(epochs,val_accuracy,'bo',label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("Training and Validation accuracy")
plt.savefig('accuracy.png')
plt.show()
plt.clf()
# loss
plt.plot(epochs,loss,'b',label='Training loss')
plt.plot(epochs,val_loss,'bo',label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title("Training and Validation loss")
plt.savefig('loss.png')
plt.show()
