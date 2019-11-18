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
parser.add_argument('base_image_path',metavar='base',type=str,
                    help='Path to the image to transform')
parser.add_argument('style_reference_image_path',metavar='ref',type=str,
                    help='Path to the style reference image')
# file preflix
parser.add_argument('result_preflix',metavar='res_preflix',type=str,
                    help='Preflix for the saved results')
args = parser.parse_args()
epoch = args.epoch
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_preflix = args.result_preflix
# weights of the diffirent loss components
style_weight = args.style_weight
content_weight = args.content_weight
tv_weight = args.tv_weight
# dimensions of the generated picture
width,height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows/height)

def preprocessing_image(image_path):
    img = load_img(image_path,target_size=(img_nrows,img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = vgg19.preprocess_input(img)
    return img


# util function to convert a tensor into a valid image
def deprocessing_image(x):
    of 
