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
from lesson4_style_transfer.utils import load_img,imshow,StylrContentModel,clip_0_1,vgg_layers
import IPython.display as display
# download images include style and content
content_path = "content.jpg"
style_path = "style1.jpg"

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
    print(' shape:',output.numpy().shape)
    print(' min:  ',output.numpy().min())
    print(" max: ",output.numpy().max())
    print(" mean :",output.numpy().mean())
    print()
# style calculate
extractor = StylrContentModel(style_layers,content_layers)
results = extractor(tf.constant(content_img))
style_results = results['style']
print("Style:")
for name , output in sorted(results['style'].items()):
    print("     ",name)
    print("     shape:",output.numpy().shape)
    print("     min:",output.numpy().min())
    print("     max:",output.numpy().max())
    print("     mean:", output.numpy().mean())
    print()
print("Content:")
for name , output in sorted(results['content'].items()):
    print("     ", name)
    print("     shape:", output.numpy().shape)
    print("     min:", output.numpy().min())
    print("     max:", output.numpy().max())
    print("     mean:", output.numpy().mean())
    print()
# gradient descent
style_targets = extractor(style_img)['style']
content_targets = extractor(content_img)['content']
# 定义一个tf.Variable 来表示要优化的图像，为了快速实现这一点，使用内容图像对其进行初始化
image = tf.Variable(content_img)

# 创建一个optimizer 推荐LBFGS 当然Adam也可以
# 这里的beta_1 and epsilon 是什么含义
optimize = tf.keras.optimizers.Adam(lr=0.02,beta_1=0.99,epsilon=1e-1)
# 为了优化，我们使用两个损失的加权组合来获得总损失
style_weight = 1e-2
content_weight = 1e4
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                           for name in content_outputs.keys()])
    loss = style_loss + content_loss
    return loss
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss,image)
    optimize.apply_gradient(zip(loss,image))
    image.assign(clip_0_1(image))
# 下面我们来运行几步
train_step(image)
train_step(image)
train_step(image)
plt.show(image.read_value()[0])

# 如果运行正常，我们来执行一个更长的优化
import time
start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.',end='')
    # display 的作用是？
    display.clear_output(wait=True)
    imshow(image.read_value())
    plt.title("Train step:{}".foamat(step))
    plt.show
end = time.time()
print("Total time :{:.1f}".format(end-start))


