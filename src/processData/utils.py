# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: Oriol Ramos Terrades
@email: oriolrt@cvc.uab.cat
'''






def generateSyntheticData( numSamples=10, numDims = 4):

    data = np.random.rand(numSamples/2,numDims)

    data = np.vstack( (data, np.sqrt(numDims/2) +  np.random.rand(numSamples/2,numDims)))

    A = dist.squareform(1/(dist.pdist(data[:,:numDims/2])+.1))
    A = A*(A>=1)

    permuta = np.random.randint(0,numSamples/2,2) + [0,numSamples/2]

    data2 = data[:,numDims/2:]
    data2[permuta,:] = data2[permuta[::-1],:]

    B = dist.squareform(1/(dist.pdist(data2)+.1))
    B = B*(B>=1)

def generateCode( method, params ):

    codeList = params.copy()
    codeList["method"] = method
    #for i, x in enumerate(paramNames):
    #    codeList[x] = str(paramValues[i])

    code = hash(frozenset(codeList.items()))

    return code #":".join(codeList)