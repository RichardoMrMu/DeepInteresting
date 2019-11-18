# -*- coding: utf-8 -*-
# @Time    : 2019-11-18 9:13
# @Author  : RichardoMu
# @File    : main1.py
# @Software: PyCharm
from wordcloud import WordCloud
from matplotlib import pyplot as plt
# english
# read file
with open("constitution.txt") as f:
    data = f.read()
# create object
wc = WordCloud().generate(data)
# show wordcloud
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()
# save file
wc.to_file('./img/en_constitution_main1.png')