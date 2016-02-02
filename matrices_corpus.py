# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:45:03 2016

@author: stuka
"""


import json
import numpy as np
import pandas as pd
from pprint import pprint
import os as os
import io as io
import sys

#os.getcwd()
#os.chdir('/home/stuka/itam2/textmining')
#os.listdir('.')

"""Metodos de la clase"""
    
"""Recalculo de probabilidades por columna"""
def normaliza_columnas_matriz(df_input):
    #print(df_input)
    for i in range(len(df_input.columns)):
        col_sum = df_input[df_input.columns[i]].sum(axis=0)
        #print(col_sum)
        col_sum = col_sum*1.0
        #print(col_sum)
        for j in range(len(df_input.index)):
            df_input.iloc[j,i] = df_input.iloc[j,i]/col_sum
    return(df_input)
    

def generaMatrices(filename='jsonCorpus2.txt'):
    
    
    
    print("Leyendo corpus")    
    """Leer el corpus"""
    data_file = io.open(filename)
    data = json.load(data_file)
    print("Leido...")    
    """
    print data  
    pprint(data)
    
    type(data)
    
    data.keys()
    data["doc"][0]
    data["doc"][0]["document"]
    data["doc"][0]["document"][0]
    data["doc"][0]["document"][1]
    data["doc"][0]["document"][0]["token"]
    value = data["doc"][0]["document"][0]["tag"]
    type(value)
    print(value)
    value == 'NP'
    len(data["doc"])
    len(data["doc"][332]["document"])
    """
    
    
    print("Extraccion de todos los tags y tokens")
    tags = []
    tokens = []
    
    for i in range(len(data["doc"])):
        for j in range(len(data["doc"][i]["document"])):
            #print "tag", data["doc"][i]["document"][j]["tag"], "token", data["doc"][i]["document"][j]["token"]
            if(data["doc"][i]["document"][j]["tag"] not in tags):
                tags.append(data["doc"][i]["document"][j]["tag"])
            if(data["doc"][i]["document"][j]["token"] not in tokens):
                tokens.append(data["doc"][i]["document"][j]["token"])
    
    
    #len(tokens)
    #len(tags)
    print("Creacion de la matriz token tags")
    df = pd.DataFrame(columns = tokens, index = tags)
    df = df.fillna(0)
    
    for i in range(len(data["doc"])):
        for j in range(len(data["doc"][i]["document"])):
           tag = data["doc"][i]["document"][j]["tag"]
           token = data["doc"][i]["document"][j]["token"]
           df.loc[tag,token] = df.loc[tag,token]+1
    df = normaliza_columnas_matriz(df)      
    
    print("Generación de la Matriz de transiciones")
    
    tag_init = u'|INIT|'
    tags.append(tag_init)
    
    dfA = pd.DataFrame(columns = tags, index = tags)
    dfA = dfA.fillna(0)
    dfA
    
    for i in range(len(data["doc"])):
        for j in range(len(data["doc"][i]["document"])):
           tag_next = data["doc"][i]["document"][j]["tag"]
           if(j-1<0):
               tag = tag_init
           else:
               tag = data["doc"][i]["document"][j-1]["tag"]
           dfA.loc[tag_next,tag] = dfA.loc[tag_next,tag] + 1
           
    #dfA.loc[tag_init,tag_init] = 1
    dfA = normaliza_columnas_matriz(dfA)
    
    print("Generacion de Iniciales")
    dfInits = pd.DataFrame(columns = [u'|INIT|'], index = tags)
    dfInits = dfInits.fillna(0)
    
    for i in range(len(data["doc"])):
           tag_inicio = data["doc"][i]["document"][0]["tag"]
           dfInits.loc[tag_inicio,u'|INIT|'] = dfInits.loc[tag_inicio,u'|INIT|'] + 1
    dfInits = normaliza_columnas_matriz(dfInits) 
    print("Devuelve tres dataframes")
    return(df,dfA,dfInits)



total = len(sys.argv)
cmdargs = str(sys.argv)
if total != 2:
    print("Usage: %s NOMBRE_ARCHIVO - debe estar en la misma carpeta del codigo" % sys.argv[0])
    sys.exit(2)
    
else:    
    result = generaMatrices(str(sys.argv[1]))
    print "Las matrices finales son", result
