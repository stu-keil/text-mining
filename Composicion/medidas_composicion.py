# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:59:19 2016

@author: stuka
"""

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import os
import io
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import pandas as pd
import sys
import gensim


def plot_words(Z,ids,mark='o',color='blue'):
	r=0
	plt.scatter(Z[:,0],Z[:,1], marker=mark, c=color)
	for label,x,y in zip(ids, Z[:,0], Z[:,1]):
		plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-1,1), textcoords='offset points', ha= 'center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.0), arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
		r+=1


os.getcwd()
os.chdir("/home/stuka/itam2/textmining/text-mining/Composicion")
os.listdir(".")

filename = 'Frases.txt'


data = [line.rstrip() for line in io.open(filename) if line.strip()]
charstosub = pd.DataFrame(zip([u'á', u'é', u'í', u'ó', u'ú',u'"',u'“',u'”',u',',u'\.',u'ñ',u'\!',u'\¡'],[u'a', u'e', u'i', u'o', u'u',u'',u'',u'',u'',u'',u'n',u'',u''])) 

for row in charstosub.iterrows():
    data = [re.sub(row[1][0],row[1][1],line) for line in data]

data = [line.lower() for line in data]   

for line in data:
    print line


data_short = [line.split('\t')[0] for line in data]
key_word = [line.split('\t')[1] for line in data]

corpus = pd.DataFrame(zip(data_short,key_word))
corpus.columns = ['oracion','clave']

list(corpus['oracion'])
sentences = list([line.split() for line in list(corpus['oracion'])])
model = gensim.models.Word2Vec(sentences, min_count=0, size=2, window=4)
words = model.vocab.keys()
for line in words:
    print line
vectors = [model[word] for word in model.vocab]
rep_vect = pd.concat([pd.DataFrame(words),pd.DataFrame(vectors)],axis=1)
rep_vect.columns = ['word','vec1','vec2']

plot_words(rep_vect[['vec1','vec2']].as_matrix(),words)







