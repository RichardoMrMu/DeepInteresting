# -*- coding: utf-8 -*-
# @Author  : RichardoMu
# use tf.data load image
import tensorflow  as tf
import os
import random
os.environ["TF_CPP_MIN_LOG_LEVEL"] ='2'
# what is this
AUTOTUNE =  tf.data.experimental.AUTOTUNE
# 下载 并检查数据集
# 搜索照片
import  pathlib
dataroot = '/home/tbw/.keras/datasets/flower_photos'
dataroot = pathlib.Path(dataroot)
# 遍历目录的子目录或者文件
for item in dataroot.iterdir():
    print(item)

# all_image_paths content all images
all_image_paths = list(dataroot.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
# check images so know what we are dealing
attributions = (dataroot/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)
import IPython.display as display
def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(dataroot)
    return  "Image (CC BY 2.0)" + " - ".join(attributions[str(image_rel)].split(' - ')[:-1])
for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
    print(caption_image((image_path)))
    print()
# 确定每张图片的标签
# 列出可用的标签
label_names = sorted(item.name for item in dataroot.glob('*/') if item.is_dir())
print(label_names)
# 为每个标签分配索引
label_to_index = dict((name,index) for index,name in enumerate(label_names))
print(label_to_index)
# 创建一个列表,包含每个文件的标签索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
print("First 10 labels indices: ", all_image_labels[:10])
# 加载和格式化图片
img_path = all_image_paths[0]
print(img_path)
# 原始数据
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+"...")
# 解码为tensor
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
# 根据模型调整大小
img_final = tf.image.resize(img_tensor,[192,192])
img_final = img_final/255.
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())
def preprocess_image(image):
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[192,192])
    image /= 255.
    return image
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

import matplotlib.pyplot as plt
image_path = all_image_paths[0]
label = all_image_labels[0]
plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(label_names[label].title())
print()

# 构建一个tf.data.Dataset
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# shape 和 types 描述数据集里每个数据项的内容,在这里是一组标量二进制字符串
print(path_ds)
# 创建一个新的数据集,
image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)
plt.figure(figsize=(8,8))
for  n ,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    plt.show()

# 创建标签数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,dtype=tf.int64))
for label in label_ds.take(10):
    print(label_names[label.numpy()])
# 将图片和标签打包在一起
image_label_ds = tf.data.Dataset.zip((image_ds,label_ds))
# 这个新的数据集的shapes和types也是维数和类型的元组
print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths,all_image_labels))
# 元组被解压缩到映射函数的位置参数中
def load_and_preprocess_from_path_label(path,label):
    return load_and_preprocess_image(path),label
image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)

# 训练
# shuffle batch repeat
batch_size = 32
# 设置一个和数据集大小一致的shuffle buffer size (随机缓冲区大小)以保证数据被充分打乱
ds = image_label_ds.shuffle(image_count).repeat().batch(batch_size)
# 当模型在训练的时候, prefetch 使数据集在后台取得batch
ds = ds.prefetch(buffer_size=AUTOTUNE)

# 传递数据集到模型
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
# 设置权重为不可训练,目的是只训练输出的全连接层
mobile_net.trainable = False
# 该模型的希望输入被标准化为[-1,1]范围内
help(keras_application.mobilenet_v2.preprocess_input)
# 在输入模型之前需要将image范围从[0,1]转化为[-1,1]
def change_range(image,label):
    return 2*image-1,label
keras_ds = ds.map(change_range)
# MobileNet 为每张图片的特征返回一个6*6的空间网格
# 传递一个batch图片 查看结果
image_batch,label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)
# 构建一个包装了MobileNet的模型并在tf.keras.layers.Dense输出层之前使用tf.keras.layers.GlobalAveragePooling2D来平均那些空间向量
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names),activation=tf.nn.softmax)
])
logit_batch = model(image_batch).numpy()
print("min logitL:",logit_batch.min())
print("max logit:",logit_batch.max())
print("shape:",logit_batch.shape)

# compile fit
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,loss=tf.losses.SparseCategoricalCrossentropy,
              metrics=[accuracy])
print(len(model.trainable_variables))
model.summary()
model.fit(ds,epochs=1,steps_per_epoch=3)