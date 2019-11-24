# -*- coding: utf-8 -*-
# @Author  : RichardoMu
# 用tf.data加载CSV数据
import functools
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
# 让numpy数据更易读
np.set_printoptions(precision=3,suppress=True)
# 加载数据
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
# !head{train_file_path}
# 从文件中读取数据并创建dataset
label_column='survived'
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name=label_column,
        na_value="?",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset
raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)
example , label = next(iter(raw_train_data))
print("example:\n",example,"\n")
print("label:\n",label)
