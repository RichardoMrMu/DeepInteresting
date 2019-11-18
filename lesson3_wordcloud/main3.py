# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 9:37
# @Author  : RichardoMu
# @File    : main3.py
# @Software: PyCharm
import jieba
from wordcloud import WordCloud
from matplotlib import  pyplot as plt
# chinese
# read file
with open("xyj.txt",'r',encoding='UTF-8') as f:
    data = f.read()
#   ch tokenization
data = ' '.join(jieba.cut(data))
print(data[:100])

# create object
wc = WordCloud(font_path='Hiragino.ttf',width=800,height=400,min_font_size=5,max_words=200,background_color=None,mode='RGBA').generate(data)
# show wordcloud
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
# save file
wc.to_file('./img/ch_xyj_main3_tokenization.png')