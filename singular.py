# -*- coding: utf-8 -*-
"""
Spyder Editor

Stephane Keil Rios
160559
"""

import os
import sys

def singular(cadena):    
        print "La cadena inicial fue: ",cadena
        if(cadena[-1]!='s'):
            if(cadena[-2:]=="ii"):
                result = cadena[:-2]+"us"
            else:
                print "No es una palabra plural"
                result = cadena
        else:
            if(cadena[-2]=='e'):
                if(cadena[-3]=='c'):
                    result = cadena[:-3]+"z"
                else:
                    result = cadena[0:len(cadena)-2]
            else:
                result = cadena[0:len(cadena)-1]
        return result


def es_vocal(letra):
    vowels = ['á', 'é', 'í', 'ó', 'ú', 'a', 'e', 'i', 'o', 'u','Á', 'É', 'Í', 'Ó', 'Ú', 'A', 'E', 'I', 'O', 'U']
    cont = vowels.count(letra)
    if cont == 1:
        return True
    else:
        return False
    

total = len(sys.argv)
cmdargs = str(sys.argv)
if total != 2:
    print("Usage: %s PALABRA_EN_PLURAL" % sys.argv[0])
    sys.exit(2)
    
else:    
    result = singular(str(sys.argv[1]))
    print "El resultado de la singularizacion es:", result






    
    
