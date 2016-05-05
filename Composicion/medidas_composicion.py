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
import math
import pandas as pd
import sys
import gensim

os.getcwd()
os.chdir("/home/stuka/itam2/textmining/text-mining/Composicion")
os.listdir(".")

filename = 'Frases.txt'


data = [line.rstrip() for line in io.open(filename) if line.strip()]
charstosub = pd.DataFrame(zip([u'á', u'é', u'í', u'ó', u'ú',u'"',u'“',u'”',u',',u'\.'],[u'a', u'e', u'i', u'o', u'u',u'',u'',u'',u'',u''])) 

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
model = gensim.models.Word2Vec(sentences, min_count=0, size=2, window=10)

model.vocab.keys()


