# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:39:54 2016

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
import math
import pandas as pd
import sys


"""Un metodo que asigna a cada elemento en label_list un id unico autoincremental"""
def dictionario(label_list):
    dict = defaultdict()
    dict.default_factory = lambda: len(dict)
    [dict[w] for w in label_list]
    return(dict)
    
def matrizPMI(filename='corpus_DSM.txt'):   
    #os.getcwd()
    #os.chdir("/home/stuka/itam2/textmining/text-mining/")
    #os.listdir(".")
    
    
    #filename = 'corpus_DSM.txt'
    
    data = [line.rstrip() for line in io.open(filename) if line.strip()]
    
    """Eliminamos caracteres indeseados"""
    charToDrop = u'[\.,;:\'\"\?¿¡\!\{\}\[\]\-_()]'
    data = [re.sub(charToDrop,'',line) for line in data]
    
    """Este era un tokenizer a patin"""
    #corpus = [line.split() for line in data]
    
    """Algorimto de stemming en español""" 
    stemmer = SnowballStemmer('spanish')
    corpus=[]
    for j in range(len(data)):
        corpus.append([stemmer.stem(i) for i in word_tokenize(data[j])])
    
    """Algoritmo de Stemming Fuerza Bruta"""
    """separadas = []
    for i in range(docs):
        palabras = documentos[i].split()
        caracteres = []
        for k in range(len(palabras)):
            palabra = palabras[k][0:5]
            caracteres.append(palabra)
        separadas.append(caracteres)"""
    
    """Extraigo los conteos de frecuencia de cada palabra en el corpus"""
    token_freq = defaultdict(int)
    count=0
    for i in corpus:
        for j in i: 
            token_freq[j] += 1   
            count += 1
    """Y lo convierto en la probabilidad de cada palabra P(wi)"""
    token_prob = {k: v*1.0 / count for k, v in token_freq.items()}
    
    
    tokens = []  
    for i in corpus:
        for j in i:
            if(j not in tokens):
                tokens.append(j)
                
    tokens = dictionario(tokens)   
    
    Matriz_PMI = np.zeros((len(tokens),len(tokens)))
    
    
    for i in range(len(corpus)):
        for j,k in tokens.iteritems():
            if(j in corpus[i]):
                for l,m in tokens.iteritems():
                    if(j!=l and l in corpus[i]):
                        Matriz_PMI[k][m] += 1
    
    Matriz_PMI = Matriz_PMI/count
    
    for j,k in tokens.iteritems():
        for l,m in tokens.iteritems():
            if((token_prob[j]*token_prob[l])==0):
                Matriz_PMI[k][m] = 0
            else:
                if(Matriz_PMI[k][m]!=0):
                    Matriz_PMI[k][m] = math.log(Matriz_PMI[k][m]/(token_prob[j]*token_prob[l]),2)
                    
                    
    PMI = pd.DataFrame(columns=sorted(tokens.keys()), index=sorted(tokens.keys()))
    
    for i in tokens.keys():
        for j in tokens.keys():
            PMI.loc[i,j] = Matriz_PMI[tokens[i]][tokens[j]]
            
    
    return(PMI)

total = len(sys.argv)
cmdargs = str(sys.argv)
if total != 2:
    print("Usage: %s NOMBRE_ARCHIVO - debe estar en la misma carpeta del codigo" % sys.argv[0])
    sys.exit(2)
    
else:    
    PMI = matrizPMI(str(sys.argv[1]))
    print "Las matrices PMI es", PMI

