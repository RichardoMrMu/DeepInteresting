# -*- coding: utf-8 -*-
# @Time    : 2019-11-20 19:53
# @Author  : RichardoMu
# @File    : deepdream.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import matplotlib as mpl
from IPython.display import clear_output
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
# url : https://tensorflow.google.cn/tutorials/generative/deepdream
# choose an image to dream-ify
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"


# download an image and read it into a numpy array
def download(url,target_size=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name,origin=url)
    img = tf.keras.preprocessing.image.load_img(image_path,target_size=target_size)
    img = np.array(img)
    return img


# normalize an image
def deprocess(img):
    img = 255*(img+1.0)/2.0
    return tf.cast(img,tf.unit8)


# display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()


# downsizing the image makes it easier to work with
# type : np.array
original_img = download(url,target_size=[255,375])
# show img
show(original_img)
# prepare the feature extraction model
# download and prepare a pre-trained image classification model . u will use
# InceptionV3 which is similar to the model originally used in DeppDream.
# Note that any pre-trained model will work,although u will have to adjust the
# layer names below if you change this

# will download data about InceptionV3
base_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

# maximize the activations of these layers
names = ['mixed3','mixed5']
layers = [base_model.get_layer(name).output for name in names]
# create the feature  extraction model
dream_model = tf.keras.Model(inputs=base_model.input,outputs = layers)

# calculate loss
# the loss is the sum of the activations in the chosen layers ,the loss is normalized
# at each layer so the contribution from large does not outweight smaller layers.Normally
# loss is a quantity you wish to minimize via gradient descent .In DeepDream, you will maximize this loss
# via gradient ascent
# attention maximize the loss

def calc_loss(img,mdoel):
    #     pass forward the image through the model to retrive the activations
    #     convert the image into a batch pf size 1
    img_batch = tf.expand_dims(img,axis=0)
    layer_activations = dream_model(img_batch)

    losses = []
    for act in layer_activations:
        loss = tf.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)
        
# gradient ascent
# once have calculate the loss for the chosen layers ,all that is left
# is to calcuulate the gradients with respect to the image,
# and add them to the original image
# adding the gradients to the image enhances the pattern seen
# by the network.At each step , u will have created an image
# that increasingly excites the activations of certain layers
# in the network
@tf.function
def deepdream(model,img,step_size):
    with tf.GradientTape() as tape:
        # this needs gradients relative to img
        # gradienttape only watches tf.Variable's by default
        tape.watch(img)
        loss = calc_loss(img,model)
    # calculate the gradient of the loss with respect to
    # the pixels of the input image
    gradients = tape.gradient(loss,img)
    # normalize the gradient
    gradients /= tf.math.reduce_std(gradients) + 1e-8

    # in gradient ascent . the loss is maximized so that the
    # input image increasingly excites the layers
    # u cna update the image by directly adding the gradients
    # because they arer the same shape
    img = img + gradients*step_size
    img = tf.clip_by_value(img,-1,1)
    return loss , img

def run_deep_dream_simple(model,img,steps=100,step_size=0.01):
    # convert from unit8 to the range expected by the model
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for step in range(steps):
        loss, img = deepdream(model,img,step_size)

        if step%100 == 0:
            clear_output(wait=True)
            show(deprocess(img))
            print("step:{},loss{}".format(step,loss))

    result = deprocess(img )
    clear_output(wait=True)
    show(result)
    return result
dream_img = run_deep_dream_simple(model=dream_model,img=original_img,
                                  steps=800,step_size=0.001)

