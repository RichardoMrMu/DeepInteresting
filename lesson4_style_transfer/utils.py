# -*- coding: utf-8 -*-
# @Time    : 2019-11-20 15:13
# @Author  : RichardoMu
# @File    : utils.py
# @Software: PyCharm
import tensorflow as tf
from matplotlib import pyplot as plt
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


# gram matrix to calculate style
def gram_matrix(input_tensor):
    # tf.linalg.einsum
    result = tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_tensor[2],dtype=tf.float32)
    return result / num_locations


class StylrContentModel(tf.keras.models.Model):
    def __init__(self,style_layers,content_layers):
        super(StylrContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    def call(self,inputs):
        """
        expects float input in [0,1]
        :param inputs:
        :return:
        """
        inputs = inputs * 255.
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs , content_outputs = (outputs[:self.num_style_layers],
                                           outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_outputs) for style_output in style_outputs]
        content_dict = { content_name : value for content_name , value in zip(self.content_layers,content_outputs)}
        style_dict = {style_name:value for style_name , value in zip(self.style_layers,style_outputs)}
        return {"content":content_dict,"style":style_dict}


# 由于这是一个浮点图像，因此我们定义一个函数来保持像素值在0到1之间
def clip_0_1(image):
    return tf.clip_by_value(image,clip_value_min=0.,clip_value_max=1.)

