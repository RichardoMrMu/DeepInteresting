# -*- coding: utf-8 -*-
# @Author  : RichardoMu
import pandas as pd
import tensorflow as tf
csv_file = tf.keras.utils.get_file("heart.csv","https://storage.googleapis.com/applied-dl/heart.csv")
df = pd.read_csv(csv_file)
print(df)
df.head()
print(df.dtypes)
df['thal'] = pd.Categorical(df['thal'])
target = df.pop('target')
print(target)