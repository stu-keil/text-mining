# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:01:55 2016

@author: stuka

Representacion vectorial con Word Embeddings
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

os.getcwd()
os.chdir("/home/stuka/itam2/textmining/text-mining/")
os.listdir(".")

filename='corpus_WE.txt'

data = [line.rstrip() for line in io.open(filename) if line.strip()]
type(data)
print(data[4])