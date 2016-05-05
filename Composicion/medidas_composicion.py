# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:59:19 2016

@author: stuka

Este programa toma un corpus:

1- Lo limpia
2- Genera una representación vectorial de todos sus token
3- Calcula de diferentes maneras la composición de palabras de cada documento para tener una representación vectorial a nivel documento
4- Utiliza un algoritmo de agrupamiento para generar conglomerados de documentos
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
    """
Este metodo permite mostrar la representación vectorial del corpus estudiado despues de aplicar word2vec    
    """
	r=0
	plt.scatter(Z[:,0],Z[:,1], marker=mark, c=color)
	for label,x,y in zip(ids, Z[:,0], Z[:,1]):
		plt.annotate(label.decode('utf8'), xy=(x,y), xytext=(-1,1), textcoords='offset points', ha= 'center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.0), arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
		r+=1

def generaClusters(matriz,numclust):
    """
    Este metodo permite generar los conglomerados de los documentos que se quieren representar, es necesario proveer una matriz de n = documentos x m = numero de dimensiones sobre la que se realiza la representación vectorial.
    """
    #### Clustering
    nclusters = 21
    modelo_cluster = KMeans(n_clusters=nclusters)
    modelo_cluster.fit(matriz)
    labels = modelo_cluster.labels_
    return labels

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
labels = generaClusters(rep_vect_doc_sum,21)

eval_final = pd.concat([corpus,pd.DataFrame(labels),pd.DataFrame(rep_vect_doc_sum)],axis=1)
eval_final.to_csv('modelo_final_sum.csv',encoding='utf-8',index=False)

##### Funcion Producto
rep_vect_doc_prod = np.full((len(corpus.index),len(rep_vect.columns)),1.0, dtype=np.double)
for line in range(len(sentences)):
    for word in range(len(sentences[line])):
        for coord in range(len(rep_vect.columns)):
            rep_vect_doc_prod[line][coord] *= rep_vect.loc[sentences[line][word]][coord]
plot_words(rep_vect_doc_prod,list(corpus['oracion']))
labels = generaClusters(rep_vect_doc_prod,21)

eval_final = pd.concat([corpus,pd.DataFrame(labels),pd.DataFrame(rep_vect_doc_prod)],axis=1)
eval_final.to_csv('modelo_final_prod.csv',encoding='utf-8',index=False)




    

