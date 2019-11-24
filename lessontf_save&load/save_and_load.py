# -*- coding: utf-8 -*-
# @Author  : RichardoMu
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def preprocess(x,y):
    # x = x.reshape(-1,28*28)
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y
# get MNIST datasets
batch_size = 32
(train_data,train_label),(test_data,test_label ) = tf.keras.datasets.mnist.load_data()
train_data = train_data[:1000].reshape(-1,28*28)
test_data = test_data[:1000].reshape(-1,28*28)
train_label = train_label[:1000]
test_label = test_label[:1000]
train_db = tf.data.Dataset.from_tensor_slices((train_data,train_label))
train_db = train_db.map(preprocess).shuffle(25000).batch(batch_size=batch_size)
test_db = tf.data.Dataset.from_tensor_slices((test_data,test_label))
test_db = test_db.map(preprocess).batch(batch_size=batch_size)
# optimizer
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
# model
def creare_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512,activation=tf.nn.relu,input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
    model.compile(optimizer=optimizer,loss=tf.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
# model = creare_model()
# model.summary()
#
# 在训练其间保存模型 以checkpoints形式保存
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
# # 创建一个保存模型权重的回调
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
# # 使用新的回调训练模型
# model.fit(train_db,epochs=10,validation_data=test_db,callbacks=[cp_callback])
# # 这可能会生成与保存优化程序状态相关的警告
# # 这额警告是防止过时使用，可以忽略

# 测试一个没有经过训练的模型
model = creare_model()
loss,acc = model.evaluate(test_db,verbose=2)
print("Untrained model ,accuracy :{:5.2f}%".format(100*acc))
# 重新加载权重
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_db,verbose=2)
print("restored model ,accuracy :{:5.2f}%".format(100*acc))