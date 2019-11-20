# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 20:18
# @Author  : RichardoMu
# @File    : vgg_train.py
# @Software: PyCharm
import argparse
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras.preprocessing.image import load_img,img_to_array
import tensorflow as tf
from keras.applications import vgg19
import numpy as np
parser = argparse.ArgumentParser(description='vgg style transfer')
parser.add_argument('--epoch',type=int,default=10,
                   help='epoch limit (default 10)')
parser.add_argument('--VGG_MODEL',type=str,default='imagenet-vgg-verydeep-19.mat',
                   help='vgg model trained')
parser.add_argument('--tv_weight',type=float,default=1.0,required=False,
                   help='Total Variation weight.')
parser.add_argument('--style_weight',type=float,default=1.0,required=False,
                   help='Style weight')
parser.add_argument('--content_weight',type=float,default=0.025,required=False,
                   help='Content weight')
# parser.add_argument('base_image_path',metavar='base',type=str,
#                     help='Path to the image to transform')
# parser.add_argument('style_reference_image_path',metavar='ref',type=str,
#                     help='Path to the style reference image')
# # file preflix
# parser.add_argument('result_preflix',metavar='res_preflix',type=str,
#                     help='Preflix for the saved results')
args = parser.parse_args()
epoch = args.epoch
# base_image_path = args.base_image_path
base_image_path = "F:\\debug\\DeepInteresting\\DeepInteresting\\lesson4_style_transfer\\content.jpg"
style_reference_image_path = "F:\\debug\\DeepInteresting\\DeepInteresting\\lesson4_style_transfer\\style1.jpg"
result_preflix = "result_preflix"
# weights of the diffirent loss components
style_weight = args.style_weight
content_weight = args.content_weight
tv_weight = args.tv_weight
# dimensions of the generated picture


def preprocessing(x):
    x = tf.io.read_file(x)
    x = tf.image.decode_image(x,channels=3)#rgba
    x = tf.image.resize(x,[800,400])
    x = tf.cast(x,dtype=tf.float32)/255.
    return x
import matplotlib.pyplot as plt
plt.imshow(preprocessing(base_image_path))
plt.grid(False)
print()

# # util function to convert a tensor into a valid image
# def deprocessing_image(x):
#     x[:,:,0] += 103.939
#     x[:,:,1] += 116.779
#     x[:,:,2] += 123.68
#     # 'bgr' -> 'rgb'
#     x = x[:,:,::-1]
#     x = np.clip(x,0,255).astype('unit8')
#     return x
#
# # get tensor representations of our images
# base_image = tf.Variable(preprocessing(base_image_path))
# style_reference_image = tf.Variable(preprocessing(style_reference_image_path))
