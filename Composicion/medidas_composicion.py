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
from sklearn.cluster import KMeans
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

#### Limpieza
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

#### Representacion Vectorial
list(corpus['oracion'])
sentences = list([line.split() for line in list(corpus['oracion'])])
model = gensim.models.Word2Vec(sentences, min_count=0, size=2, window=4)
words = model.vocab.keys()
for line in words:
    print line
vectors = [model[word] for word in model.vocab]
rep_vect = pd.DataFrame(vectors,index=words)
rep_vect.columns = ['vec1','vec2']

plot_words(rep_vect.as_matrix(),words)

##### Funcion Suma
rep_vect_doc_sum = np.full((len(corpus.index),len(rep_vect.columns)),0.0, dtype=np.double)
for line in range(len(sentences)):
    #print line, sentences[line]
    
    for word in range(len(sentences[line])):
        #print word, sentences[line][word]
        for coord in range(len(rep_vect.columns)):
            rep_vect_doc_sum[line][coord] += rep_vect.loc[sentences[line][word]][coord]
 
plot_words(rep_vect_doc_sum,list(corpus['oracion']))

corpus['clave'].value_counts()
corpus['clave'].value_counts().count()

#### Clustering
nclusters = 21
modelo_cluster = KMeans(n_clusters=nclusters)
modelo_cluster.fit(rep_vect_doc_sum)
labels = modelo_cluster.labels_
eval_final = pd.concat([corpus,pd.DataFrame(labels),pd.DataFrame(rep_vect_doc_sum)],axis=1)
eval_final.to_csv('modelo_final.csv',encoding='utf-8',index=False)
