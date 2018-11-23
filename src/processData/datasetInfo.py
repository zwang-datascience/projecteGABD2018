# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: Oriol Ramos Terrades
@email: oriolrt@cvc.uab.cat
'''


class datasetInfo(object):
  """
  Conté la informació necessaria per manipular els datasets dels experiments

  """

  def __init__(self, name="", tipus=""):
    '''
        Constructor
        '''
    self.__name = name
    if tipus.lower() in ["vector", "image"]:
      self.__type = tipus
    else:
      self.__type = "unknown"
    self.__features = ""
    self.__classIds = []
    self.__classList = []
    self.__layers = ""

  def __str__(self):
    sortida = " {}".format(self.__name)
    if self.__type == "image":
      sortida = sortida + " cnn {} layer {}".format("AlexNet",self.__layers)

    return sortida
  @property
  def name(self):
       return self.__name

  @name.setter
  def name(self, name):
    self.__name = name

  @property
  def type(self):
       return self.__type

  @type.setter
  def type(self, valor ):
    if valor.lower() in ["vector", "image"]:
      self.__type = valor
    else:
      self.__type = "unknown"


  @property
  def features(self):
       return self.__features

  @features.setter
  def features(self, valor):
    self.__features = valor

  @property
  def classIds(self):
       return self.__classIds

  @classIds.setter
  def classIds(self, valor ):
      self.__classIds = valor

  @property
  def classList(self):
       return self.__classList

  @classList.setter
  def classList(self, valor ):
      self.__classList = valor

  def __getitem__(self, item):
    return self.__getattribute__(item)

  def __setitem__(self, key, value):
    self.__setattr__(key, value)

  #
  # @property
  # def setFeatures(self, features):
  #     self._features = features
  #
  # @property
  # def setClassIds(self, classIds):
  #     self._ClassIds = classIds

