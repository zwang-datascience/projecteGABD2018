'''
Created on Jun 12, 2018

@author: oriolrt
'''

#from scipy.sparse import csgraph
#from scipy.sparse.linalg import svds
#from sklearn.cluster import KMeans

#import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np
#import scipy.sparse as sp

import random
import struct
from collections import namedtuple


class OutlierDetector(object):
    '''
    It implements a base class for outlier detectors
    '''


    def __init__(self, numViews=2):
        '''
        Constructor
        '''
        
        self.numViews = 2
        self.eps = 10**-6
        self.rho = 1.2
        self.mu_max = 10**6
        self.mu=10**-1
        self.maxIter = 200
        self.setViews( numViews )


    def setViews(self, numViews):
        self.numViews = numViews
        self.layers = ['Visual']*numViews

    def __generateClassOutlier(self,data, IdsList, ratio=0.05):

        # numViews = len(data)

        # keys = IdsList.keys()
        len_class = [len(IdsList[x]) for x in IdsList]
        total = sum(len_class)

        if isinstance(ratio, list):
            numOutliers = len(ratio)
            rid = np.array(ratio)


        else:
            numOutliers = int(round(total * ratio / 2.0))
            idClasses = random.sample(range(len(len_class)), 2)
            rid = np.zeros((numOutliers, len(len_class)))

            for i, l in enumerate(IdsList):
                rid[:, i] = random.sample(IdsList[l], numOutliers)

            rid = rid.astype(int)

        fila = namedtuple("fila", "id features")

        outliersGTIdx = []
        for i in range(numOutliers):
            currentView = 0  ##random.randint(0,numViews-1)
            idClasses = random.sample(range(len(len_class)), 2)
            features = data[currentView][rid[i, idClasses[0]]].features
            data[currentView][rid[i, idClasses[0]]] = fila(id=data[currentView][rid[i, idClasses[0]]].id,
                                                           features=data[currentView][rid[i, idClasses[1]]].features)
            data[currentView][rid[i, idClasses[1]]] = fila(id=data[currentView][rid[i, idClasses[1]]].id,
                                                           features=features)
            outliersGTIdx = outliersGTIdx + rid[i, idClasses].tolist()
            # res.append( idClasses )

        outliersGTIdx.sort()

        # y = np.zeros(total)
        # y[outliersGTIdx] = 1

        return data, outliersGTIdx


    def prepareExperimentData(self,dbms, conf, nameDataset, dataInfo, repeticio, settings = {'numberOfViews': 2} ):
        """
        Generates the outlier data given the settings for outlier Detection methods.

        :param features: list of feature vectors used to geneta
        :param classIds:
        :param settings:
        :return:
        """


        class_outlier =  float(conf[0] / 100.0)
        attr_outlier =  float(conf[1] / 100.0)
        if "numberOfViews" in settings:
            numViews = settings["numberOfViews"]
        else:
            numViews = 2


        if len(dataInfo.features) == 1: dataInfo.features= dataInfo.features[0]
        numDims = len(dataInfo.features[0].features)
        numSamples = len(dataInfo.features)
        numFeatures = int(numDims / numViews)
        newFeatures = [[]]*numViews



        fila = namedtuple("fila", "id features")
        maxVal=[-1000000]*numViews
        for x in dataInfo.features:
            id=x.id

            for y in range(numViews):
                feats = x.features[y*numFeatures:(y+1)*numFeatures]
                newFeatures[y] = newFeatures[y] + [fila(id=id, features=feats)]


        outliers,generateOutliersFlag =  dbms.loadOutliers(nameDataset, repeticio, numSamples,conf, dataInfo )

        if generateOutliersFlag:
            newFeatures, outliersGTIdx = self.__generateClassOutlier(newFeatures,dataInfo["classIds"],ratio=class_outlier)

            oid = list(set(range(numSamples)).difference(outliersGTIdx))
            #random.shuffle(id)

            if isinstance( attr_outlier, list):
                mostres = attr_outlier
            else:
                N = int(numSamples*attr_outlier)
                mostres = random.sample(oid, N)

            attrOut = []
            for x in mostres:
                attrOut.append(x)
                for i in range(numViews):
                    row = newFeatures[i][x]
                    newFeatures[i][x] = fila(id=row.id, features=np.hstack(np.random.rand( 1,numFeatures )).tolist() )

            outliersGTIdx = outliersGTIdx + attrOut


            #Salvem els outliers en la BD
            dbms.insertOutlierData(newFeatures,nameDataset, repeticio ,outliersGTIdx, conf, dataInfo)

        else:
            outliersGTIdx = outliers.keys()

            #separem els vectors en les vistes
            for oid in outliers:
                #sprint(oid)
                for y in range(numViews):
                    newFeatures[y][oid] = fila(id=oid, features=outliers[oid][y*numFeatures:(y+1)*numFeatures])

        y = np.zeros(numSamples)
        y[outliersGTIdx] = 1

        return newFeatures, y, outliersGTIdx



    
