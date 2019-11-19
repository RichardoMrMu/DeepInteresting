# -*- coding:utf-8 -*-
# @Time     : 2019-11-17 20:39
# @Author   : Richardo Mu
# @FILE     : style_transfer.PY
# @Software : PyCharm
# https://tensorflow.google.cn/tutorials/generative/style_transfer
# use code in tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import time
import functools

# download images include style and content
content_path = "1.jpg"
style_path = "2.jpg"
def load_img(path_to_img):
    # max size is 512
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_imagei(img,channels=3)
    img = tf.image.convert_image_dtype(img,dtype=tf.float32)

    shape = tf.cast(tf.shape(img)[:-1],tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape*scale,tf.int32)
    img = tf.image.resize(img,new_shape)

    img = img[tf.newaxis,:]
    return img
def imshow(image,title=None):
    if len(image.shape)>3:
        image = tf.squeeze(image,axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(included_top=False,weights='imagenet')
    vgg.trainable = True
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input],outputs)
    return model

# show img has downloaded
content_img = load_img(content_path)
style_img = load_img(style_path)
plt.subplot(1,2,1)
imshow(content_img,'content_img')
plt.subplot(1,2,2)
imshow(style_img,'style_img')

# load vgg
img_resize = 224
x =tf.keras.applications.vgg19.preprocess_input(content_img*255)
x = tf.image.resize(x,(img_resize,img_resize))
vgg = tf.keras.applications.VGG19(include_top=True,weights='imagenet')
prediction_probabilities = vgg(x)
print(prediction_probabilities.shape)
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
print([(class_name,prob) for (number,class_name,prob) in predicted_top_5])

vgg = tf.keras.applications.VGG19(included_top=False,weights='imagenet')
for layer in vgg.layers:
    print(layer.name)
# get layers output to represent style and content
# content layer feature maps
content_layers = ["block5_conv2"]
# style layers
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
# build model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_img*255)
# show every layer's infomation
for name,output in zip(style_layers,style_outputs):
    print(name)
    print(' shape:',)

