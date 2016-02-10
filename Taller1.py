# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:04:52 2016

@author: stuka
"""
from __future__ import division
import os
from json import load
from sys import argv
from collections import Counter, defaultdict
from pickle import dump
import numpy as np


os.getcwd()
os.chdir('/home/stuka/itam2/textmining/text-mining')
os.listdir('.')

filename = 'jsonCorpus2.txt'
data_file = io.open(filename)
data = json.load(data_file)




#Clase de mineria de textos
f = load(open(argv[1],'r'))["doc"]
f = data["doc"] #Para poder hacer pruebas en interactivo
docs = [d["document"] for d in f]
docs

chain = []
obs = []
S_f = Counter()
O_f = Counter()
Pi_f = Counter()
for e in docs:
    Pi_f[e[0]["tag"]] += 1
    for word in e:
        tag = word["tag"]
        token = word["token"]
        S_f[tag] += 1
        O_f[token] += 1
        chain.append(tag)
        obs.append((token,tag))

len(chain)
len(obs)
        
def voc():
    dict = defaultdict()
    dict.default_factory = lambda: len(dict)
    return dict
    
def get_ids(C, dict):
    yield [dict[w] for w in C]

S_v = voc()
chain_num = list(get_ids(chain, S_v))[0]

O_v = voc()
list(get_ids(O_f.keys(),O_v))[0]



n = len(S_v)
m = len(O_v)

def pr(x,cond,n, l=1.0): #l es el smoothing factor
    return((x+l)/(cond + l*n))
    
A = np.zeros((n,n))
B = np.zeros((m,n))
Pi = np.zeros(n)

for t in S_v.keys():
    Pi[S_v[t]] = pr(Pi_f[t],len(docs),n)

t_t1 = Counter(zip(chain,chain[1:]))    
for (t,t_1), f_t in t_t1.iteritems():
    i= 0
    for ta, ta_1 in t_t1.keys():
        if t == ta:
            i+=1
    A[S_v[t],S_v[t_1]] = pr(f_t,S_f[t],i)
    
for i in A:
    print(sum(i))

c_SO = Counter(obs)
print(O_f[u'Buda_Gautama_De_Wikipedia'])
for o in O_f.keys():
    for s in S_v.keys():
        B[O_v[o],S_v[s]] = pr(c_SO[(o,s)],O_f[o],n)
B[O_v[u'Buda_Gautama_De_Wikipedia']]
for i in B:
    print(sum(i))

out = open('HMM.p','w')
HMM = [A,B,Pi,dict(S_v),dict(O_v)]
dump(HMM,out)