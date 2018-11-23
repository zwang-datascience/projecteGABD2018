# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: oriolrt
'''

import glob
import os
import re
from collections import namedtuple
from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
import struct

import numpy as np
import utils as u


class noConnexion(object):
  """
  This class emulates a ficticious connection to a DBMS it provides the basic interface.

  """

  def __init__(self,cfg=""):
    '''
        Constructor
        '''

    self.__cfg = cfg
    self.__conn = 0
    self.__bd = ""
    self.__isStarted = False

  @property
  def bd(self):
    if not self.__isStarted:
      self.startSession()
      self.__isStarted = True

    return self.__bd

  @bd.setter
  def bd(self, nameBD):
    self.__bd = nameBD

  def __getitem__(self, item):
    return self.__getattribute__(item)

  def __setitem__(self, key, value):
    self.__setattr__(key, value)




  @property
  def connectDB(self):
    """
      Connect to a DBMS server given the connexion information saved on the cfg member variable.

      :return: None
    """

    print("""Ara ens estariem conectant al servidor...""")

    return 0


  def close(self):
    self.__isStarted = False

  def exists(self, dataset):
      """
      :param dataset: name of the dataset
      :return: True if all the feature vectors of the dataset are already inserted in the DB
      """

      return False

  def getDatasetType(self,nameDataset="vector"):
    return nameDataset

  def insertVectorDataset(self, nameDataset, fileName, params ):
    """
        Inserts the contents stored in fileName into de DB

        Params:
        :param nameDataset: name of the dataset to be inserted into the database.
        :param fileName: full path of the file where the data to be imported is stored.
        :param params: dataset params, read from the config file.
        :return: None
    """



  def insertImageDataset(self, dataset, fileName, params, labels='anno', imageExt='.jpg', labelsExt='.txt'):
    """
      This method imports the information related to image datasets. It has to insert all the data before to commit changes.

      :param dataset: name of the dataset to be imported into the DBMS .
      :param fileName: full path of the file where the feature vectors are stored.
      :param params: params read from the config file (if required).
      :param labels: name of the folder where label files are found.
      :param imageExt: images extension
      :param labelsExt: file labels extension
      :return: True if the image dataset is succesfully imported into the DB.
    """


    dataDir = "/".join(os.path.dirname(fileName).split("/")[0:-1])
    dirImages = dataDir + "/images"




    print ("Inserted info image dataset {!r}".format(dataset))
    return True


  def insertDescriptors(self, dataset, featuresName, fileName, params, featuresDir='features'):
      """

      :param dataset:
      :param features:
      :param fileName:
      :param params:
      :param featuresDir:
      :return:
      """

      """
      El següent codi llegeix els fitxers de caracteristiques i els processa. Haureu de decidir com ho guardeu en la 
      vostra BD.
      """

      dataDir = "/".join(os.path.dirname(fileName).split("/")[0:-1])
      dirImages = dataDir + "/" + featuresDir


      featureSchema, fileExt = os.path.splitext(fileName)
      if fileExt[-8:].lower() != 'features':
          print ("Warning {} not seems to be a feature file".format(fileName))
      layer = fileExt[1:-8]





      print ("""{} features of scheme {} correctly inserted """.format(layer, featureSchema))

  def insertExperiment(self, conf, repeticio,  method, paramsMethod):
    '''
    Aquesta funció guarda en la BD la informació que identifica un experiment:

    :param conf: ratio of class outliers and attribute outliers
    :param method: name of the evaluated outlier detection algorithm
    :param paramsMethod: list of parameter names
    :return: Experiment ID


    '''


    """inserim la informació dels experiments"""
    print("""A entrat en la funció insertExperiment. 
    Aquesta funció guarda en la BD la informació que identifica un experiment:
    + configuració: {}
    + repeticio: {}
    + mètode: {}
    + paràmetres del mètode: {}""".format(conf, repeticio, method, paramsMethod))



    return repeticio


  def insertOutlierData(self, newFeatures, nameDataset, repeticio, outliersGTIdx, conf , dataInfo ):
    print("""A entrat en la funció insertOutlierData. 
    Això vol dir que en la BD no hi ha outliers pels experiments identificats amb els paràmetres:
    + nom dataset: {}
    + configuració: {}
    + repeticio: {} """.format(nameDataset, conf, repeticio))


  def insertResults(self, nameDataset, idExperiment, fpr, tpr, auc, dataInfo):
    print("""A entrat en la funció insertResults. 
    Aquesta funció guarda en la BD la informació que identifica els resultats d'un experiment:
    + nom del Dataset: {}
    + id del Experiment: {} (obtingut de la crida a la funció insertExperiment)
    """.format(nameDataset, idExperiment))


  def loadOutliers(self, nameDataset, repeticio, numSamples, conf, dataInfo):
    """

    :param nameDataset:
    :param repeticio:
    :param numSamples:
    :param conf:
    :param dataInfo:
    :return:
    """

    numTotalOutliers = int(2 * round(conf[0] / 100.0 / 2.0 * numSamples)) + int(round(conf[1] / 100.0 * numSamples))

    outliers = {}

    generateOutliersFlag = numTotalOutliers != len(outliers)

    print("""Ara hauriem de accedir a la BD i mirar de recuperar els outliers si ja existeixen pel dataset {} i la 
    repeticio {}. En cas que no hi hagi els outliers necessaris retornarà FALS """.format(nameDataset,repeticio))

    return outliers,generateOutliersFlag



  def __loadVectorData(self, dataInfo):
    """
    Genera dades sintetiques donades pels parametres de l'inici de la funcio
    :param dataInfo: No s'usa
    :return: taula: llista de vectors de caracteristiques,
    :return: ids: diccionari amb les etiquetes de les classes i una llista de les ids dels vectors que pertanyen a cada classe
    """


    numSamples = 1000
    numDims = 6
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

  def loadData(self,nameDataset, dataInfo):
    return self.__loadVectorData(dataInfo)

  def startSession(self):
    self.__isStarted = True

    return True

  def commit(self):
    pass

