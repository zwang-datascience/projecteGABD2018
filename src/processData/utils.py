# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: Oriol Ramos Terrades
@email: oriolrt@cvc.uab.cat
'''


import numpy as np
from collections import namedtuple



def generateSyntheticData( numSamples=1000, numDims = 6):


    numClusters = 2

    fila = namedtuple("fila", "id features")

    ids = {}
    inici = 0
    for j in range(0, numClusters):
      ids[str(j)] = range(inici, inici + numSamples / numClusters)
      inici = inici + numSamples / numClusters
      data = np.random.rand(numSamples / numClusters, numDims)
      data = np.vstack((data, j * np.sqrt(numDims) + np.random.rand(numSamples / numClusters, numDims)))

    # creem l'estructura tabular per emular el resultat d'una query en una BD
    taula = []
    for id in range(0, numSamples):
      tupla = fila(id=id, features=data[id, :].tolist())
      taula.append(tupla)


    return taula, ids

def generateCode( method, params ):

    codeList = params.copy()
    codeList["method"] = method
    #for i, x in enumerate(paramNames):
    #    codeList[x] = str(paramValues[i])

    code = hash(frozenset(codeList.items()))

    return code #":".join(codeList)