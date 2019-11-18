# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 9:24
# @Author  : RichardoMu
# @File    : main2.py
# @Software: PyCharm

from wordcloud import WordCloud
from matplotlib import pyplot as plt
# chinese
# read file
with open("xyj.txt",'r',encoding='UTF-8') as f:
    data = f.read()
# create object
wc = WordCloud(font_path='Hiragino.ttf',width=800,height=400,min_font_size=5,max_words=20,background_color=None,mode='RGBA').generate(data)
# show wordcloud
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
# save file
wc.to_file('./img/ch_xyj_main2.png')