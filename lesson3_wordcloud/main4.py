# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 9:43
# @Author  : RichardoMu
# @File    : main4.py
# @Software: PyCharm
# use mask
import jieba
from wordcloud import WordCloud
from matplotlib import  pyplot as plt
import numpy as np
from PIL import Image
# chinese
# read file
with open("xyj.txt",'r',encoding='UTF-8') as f:
    data = f.read()
#   ch tokenization
data = ' '.join(jieba.cut(data))
print(data[:100])

# create object
# attention mask type is np.array
mask = np.array(Image.open('black_mask.png'))
wc = WordCloud(font_path='Hiragino.ttf',mask=mask,width=800,height=400,min_font_size=5,max_words=200,background_color=None,mode='RGBA').generate(data)
# show wordcloud
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
# save file
wc.to_file('./img/ch_xyj_main4_tokenization_mask.png')