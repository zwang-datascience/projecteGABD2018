# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: oriolrt
'''

import glob
import numpy as np
import os
import re
import struct
from collections import namedtuple
from os.path import expanduser

import cx_Oracle
from sshtunnel import SSHTunnelForwarder

import utils as u
import noConnection as nc


class oracleConnexion(nc.noConnexion):
  '''
    classdocs
    '''

  def __init__(self, cfg):
    '''
        Constructor
        '''
    super(oracleConnexion, self).__init__(cfg)
    self.__size = 8
    self.__typeFloat = 'd'
    self.__cursor = 0

  @property
  def cursor(self):
    if not self._noConnexion__isStarted:
      self.connectDB
      self._noConnexion__isStarted = True

    return self.__cursor

  @cursor.setter
  def cursor(self,cursor):
    self.__cursor = cursor

  @property
  def typeFloat(self):
    return self.__typeFloat

  @typeFloat.setter
  def typeFloat(self, tipus):
    mida = { 'd' : 8 , 'f' : 4}
    if tipus.lower() in ['d' ,'f']:
      self.__size = mida[tipus.lower()]
      self.__typeFloat = tipus.lower()
    else:
      self.__size = mida['d']
      self.__typeFloat = 'd'

  @property
  def size(self):
    return self.__size


  @property
  def connectDB(self):
    """
      Connect to a oracle server given the connexion information saved on the cfg member variable.

      :return: None
    """
    cfg = self._noConnexion__cfg
    numServers = len(cfg.hosts)

    if numServers == 1:
      host = cfg.hosts[0]

      if cfg.password == "":
          cfg.password  = raw_input("Password de l'usuari {} d'Oracle: ".format(cfg.username))


      if "ssh" in host:
        sshParams = host["ssh"]

        DSN =  "{}/{}@localhost:{}/{}".format(cfg.username,cfg.password,sshParams["port"],cfg.sid)

        if "password" in sshParams:
          if sshParams["password"] == "":
            sshParams["password"] = raw_input("Password de l'usuari {} a {}: ".format(sshParams["username"], host["hostname"]))
        else:
          sshParams["password"] = raw_input("Password de l'usuari {} a {}: ".format(sshParams["username"], host["hostname"]))


        self.server = SSHTunnelForwarder((sshParams["hostname"], int(sshParams["port"])),
                                    ssh_username=sshParams["username"],
                                    ssh_password=sshParams["password"],
                                    remote_bind_address=(host["hostname"], host["port"]),
                                    local_bind_address=("", int(sshParams["port"]))
                                    )

        self.server.start()


      else:
        DSN = "{}/{}@{}:{}/{}".format(cfg.username, cfg.password, host["hostname"], host["port"],
                                      cfg.sid)

      self._noConnexion__conn = cx_Oracle.connect(DSN)
      self.__cursor = self._noConnexion__conn.cursor()

    else:
      print ("Only one server connexion is allowed. Check the config file and run the script again.")
      exit(-1)

    return self._noConnexion__conn

  def close(self):
    self._noConnexion__conn.close()
    if hasattr(self, 'server'): self.server.stop()


  def commit(self):
      self._noConnexion__conn.commit()

  def exists(self, dataset):
    """

    :param dataset: name of the dataset
    :return: True if  the dataset is already inserted in the DB
    """
    #TODO:
    # + Feu la consulta que us permeti determinar si ja heu inserit abans les dades o no. Podeu utilitzar el codi
    # d'exemple que hi ha a continuació

    query = "select  ... '{}'".format(dataset)



    #return self.cursor.execute(query).fetchone() is not None
    return False

  def getDatasetType(self,nameDataset):
    """
    Retorna el tipus de conjunt de dades, si es tracta de tipus 'vector' o tipus 'image'.

    OBSERVACIO: Aquesta funcio es pot ignorar

    :param nameDataset:
    :return:
    """


    #TODO: Substituir aquest codi per un acces a la BD que retorni el tipus de datset: vector o image.
    type = { 'iris': "vector" , 'letter-recognition': "vector", 'breast-cancer' : "vector" }


    # Caldria executar un codi d'aquest estil
    #type = self.cursor.execute("""select type
    #  from  ...
    #  where  .... name = '{}' ... """.format(nameDataset)).fetchone()

    if nameDataset.lower() in type:
      return type[nameDataset.lower()]
    else:
      return "unknown"




  def __loadVectorData(self, nameDataset, classList):
    '''
    Llegeix els vectors de característiques dels datasets de vector de la BD i les guarda en una llista 'taula'. ids és un diccionari on les claus
    són les classes del dataset i els valors un llistat amb les ids de les mostres

    :param nameDataset:
    :param classList:
    :return:
    '''

    res = self.cursor.callfunc("gabd.loadVectorData", cx_Oracle.CURSOR, [nameDataset])

    fila = namedtuple("fila", "id features")


    taula = []
    ids = {}
    for row in res:
        bf = row[2].read()
        numFeatures = len(bf) / self.size
        taula.append(fila(id=row[1], features=list(struct.unpack(self.typeFloat * numFeatures, bf))))
        if row[0] in ids.keys():
            ids[row[0]] = ids[row[0]] + [row[1]]
        else:
            ids[row[0]] = [row[1]]

    return taula, ids

  def __loadImageData(self, nameDataset, params={}):
    '''
    Llegeix els vectors de característiques dels datasets d'imatges de la BD i les guarda en una llista 'taula'.
    Cal indicar el nom de la xarxa i la capa.

    ids és un diccionari on les claus són les classes del dataset i els valors un llistat amb les ids de les mostres

    :param nameDataset:
    :param params:
    :return:
    '''

    if "layers" in params:
      layers = params["layers"]
    else:
      layers = []

    if "classList" in params:
      classList = params["classList"]
    else:
      classList = []

    res = self.cursor.callfunc("gabd.loadImageData", cx_Oracle.CURSOR, [nameDataset, cnn, layers, classList])


    fila = namedtuple("fila", "id features")
    taula = []
    ids = []


    return taula, ids

  def loadData(self,nameDataset, data):

    if data.type == "vector":
      features, classIds = self.__loadVectorData(nameDataset, data)

    if data.type == "image":
        features, classIds = self.__loadImageData(nameDataset, data)

    return features, classIds


  def __loadVectorOutlier(self,nameDataset, conf, repeticio ):
    return self.cursor.callfunc("gabd.loadVectorOutliers", cx_Oracle.CURSOR,
                               [nameDataset, "-".join([str(x) for x in conf]), repeticio])



  def __loadImageOutlier(self,nameDataset, conf, repeticio, dataInfo):
    return self.cursor.callfunc("gabd.loadImageOutliers", cx_Oracle.CURSOR,
                               [nameDataset, "-".join([str(x) for x in conf]) , repeticio, dataInfo.cnn, dataInfo.layers])

    pass

  def loadOutliers(self, nameDataset, repeticio, numSamples, conf, dataInfo):
    """
    Carrega els outliers

    :param nameDataset:
    :param repeticio:
    :param numSamples:
    :param conf:
    :param dataInfo:
    :return:
    """

    numTotalOutliers = int(2 * round(conf[0] / 100.0 / 2.0 * numSamples)) + int(round(conf[1] / 100.0 * numSamples))

    if dataInfo.type == "vector":
      res = self.__loadVectorOutlier(nameDataset, conf, repeticio )

    if dataInfo.type == "image":
      res = self.__loadImageOutlier(nameDataset, conf, repeticio, dataInfo)


    outliers = {}
    for row in res:
      bf = row[1].read()
      numFeatures = len(bf) / self.size
      outliers[row[0]] = list(struct.unpack(self.typeFloat * numFeatures, bf))

    generateOutliersFlag = numTotalOutliers != len(outliers)



    return outliers,generateOutliersFlag

  def __insertImageOutliers(self, nameDataset, conf, repeticio, outliersGTIdx, vectorBlobs,dataInfo):
    self.cursor.callproc("gabd.insertImageOutliers",
                         [nameDataset, "-".join([str(x) for x in conf]), repeticio, dataInfo.cnn,
                          dataInfo.layers, outliersGTIdx, vectorBlobs])




  def __insertVectorOutliers(self,nameDataset, conf, repeticio,outliersGTIdx,vectorBlobs,dataInfo):
    for i,id in enumerate(outliersGTIdx):
      self.cursor.callproc("gabd.insertVectorOutliers", [nameDataset, "-".join([str(x) for x in conf]), repeticio, id,vectorBlobs[i]])

    self.commit()


  def insertOutlierData(self, newFeatures, nameDataset, repeticio, outliersGTIdx, conf , dataInfo ):
    """

    :param newFeatures:
    :param nameDataset:
    :param repeticio:
    :param outliersGTIdx:
    :param conf:
    :param dataInfo:
    :return:
    """

    numViews = len(newFeatures)

    vectorBlobs = [self.cursor.var(cx_Oracle.BLOB)] * len(outliersGTIdx)
    for i,id in enumerate(outliersGTIdx):
      vectorBlobs[i].setvalue(0, np.hstack([newFeatures[y][id].features for y in range(numViews)]).tobytes())


    if dataInfo["type"] == "vector":
      self.__insertVectorOutliers(nameDataset, conf, repeticio,outliersGTIdx,vectorBlobs,dataInfo)

    if dataInfo["type"] == "image":
      self.__insertImageOutliers(nameDataset, conf, repeticio,outliersGTIdx,vectorBlobs,dataInfo)

    pass

  def insertDescriptors(self, dataset, featuresName, fileName, params, featuresDir='features'):
    """

    :param dataset:
    :param featuresName:
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
    layer = ""
    cnn = ""

    featureSchema, fileExt = os.path.splitext(fileName)
    if fileExt[-8:].lower() != 'features':
      print("Warning {} not seems to be a feature file".format(fileName))
    layer = fileExt[1:-8]

    # load feature vector
    f = open(fileName, 'r')
    dims = [int(val) for val in f.readline().rstrip("\n").split(" ")]
    x = f.read()
    f.close()
    size = int(params['feat_size'])  # bytes to represent each float
    type = params['type']

    for i in range(0, dims[0]):
      # cur.execute(None, {'image_id': i, 'layer': layer, 'features': Features[i] })
      bf = x[i * dims[1] * size:(i + 1) * size * dims[1]]
      numFeatures = len(bf) / size
      features = list(struct.unpack(type * numFeatures, bf))

      # TODO: inseriu les dades en la collecció



    print """{} features of CNN {} correctly inserted """.format(layer, cnn)


  def insertRepeticio(self, conf, i=0):
    """
      Inserim en la taula Experiment la informació bàsica de l'experiment, si no existeix

      :param conf: tipus d'experiment. string de la forma <class_outlier>-<attr_outlier>
      :param i: numero d'iteracio
      :return: True o False en funció si les dades s'han inserit, o no.
     """

    #TODO: A implementar en la sessió d'ORACLE


    idRepeticio = 0
    return idRepeticio


  def insertVectorDataset(self, nameDataset, fileName, params ):
    """
      Inserts the contents stored in fileName into de DB

      Params:
      :param nameDataset: name of the dataset to be inserted into the database.
      :param fileName: full path of the file where the data to be imported is stored.
      :param params: dataset params, read from the config file.
      :return: None
    """
    #
    cur = self.cursor

    datasetParams = params['datasets'][nameDataset.lower()]
    if "label_pos" in datasetParams:
      label_pos = int(datasetParams['label_pos'])
    else:
      label_pos = -1

    # insert dataset

    """
    Comprovem si el nombre de classes del dataset està en params o no
    """
    if "k" in params:
      # TODO: Haureu de guardar aquesta informació a la BD
      print("k: {}".format(params["k"]))
      k = params["k"]
    else:
      # TODO: Haureu de guardar la informació de la BD sense la inforamció de la K (numero de classes)
      print("Haureu de guardar la informació de la BD sense la inforamció de la K (numero de classes)")

    """
    Anem a llegir el contingut dels fitxers i els guardem en una estructura definida per fila amb camps: id, label, feature
    """
    fila = namedtuple("fila", "id label feature ")
    f = open(fileName)
    line = f.readline().rstrip("\n").rstrip("\r")

    features = []
    id = 0
    labels = []
    while line:
      stringFeatures = line.split(",")
      label = stringFeatures[label_pos]
      stringFeatures.remove(label)
      feature = [float(x) for x in stringFeatures]
      if len(feature) > 0:
        features.append(fila(id=id, label=label, feature=feature))
        id = id + 1

      line = f.readline()
    f.close()

    """
    Caldra guardar la informació a la BD segons l'estructura que hagueu decidit
    """
    for row in features:
      # TODO: insereu les dades
      feature = row.feature


    self.commit()


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

    # TODO: Haureu d'acabar d'implementar aquesta funció més endevant
    #

    cur = self.cursor

    # insert dataset
    dataDir = "/".join(os.path.dirname(fileName).split("/")[0:-1])
    dirImages = dataDir + "/images"

    """
    El següent bucle recorre la carpeta de imatges i extreu la id de la imatge a partir del nom
    """
    listImages = glob.glob(dirImages + '/*' + imageExt)
    for imageName in listImages:
      nom, _ = os.path.splitext(os.path.basename(imageName))
      match = re.search(r'im(?P<id>\d+)', nom)

      data = {'name': nom, 'image_id': int(match.group('id'))}

      # TODO: caldra inserir la informació en la BD

    # load image annotation
    wd = dataDir + '/' + labels + '/'
    fileLabels = glob.glob(wd + '*' + labelsExt)

    """
    El següent codi recorre la carpeta amb els fitxers d'anotacions i recupera la informació de les etiquetes
    """
    for fileLabel in fileLabels:
      nom, _ = os.path.splitext(os.path.basename(fileLabel))
      if nom[-3:] != '_r1' and nom != "README":
        f = open(fileLabel)
        line = f.readline().rstrip("\n").rstrip("\r")
        while line:
          if len(line) > 0:
            data = {'label': nom, 'image_id': int(line)}

            # TODO: caldrà inserir les dades a la BD

            line = f.readline().rstrip("\n").rstrip("\r")

        f.close()

    self.commit()

    print("Inserted info image dataset {!r}".format(dataset))
    return True




  def insertExperiment(self, conf, repeticio,  method, paramsMethod):
    '''
    Inserim en la taula Experiment la informació bàsica de l'experiment, si no existeix

    :param conf: ratio of class outliers and attribute outliers
    :param method: name of the evaluated outlier detection algorithm
    :param paramsMethod: list of parameter names
    :return: None


    '''

    """
    Per identificar de manera única els conjunts de paràmetres (i valors) associats a cada experiment generem un codi
    """

    #paramNames = paramsMethod.keys()
    #paramValues = paramsMethod.values()
    paramCode = u.generateCode(method, paramsMethod )

    idEM =self.cursor.callfunc("GABD.insertExperiment", int, ["-".join([str(x) for x in conf]), conf[0], conf[1], repeticio, method, paramCode,  paramsMethod.keys(), [ float(x) for x in paramsMethod.values()] ])

    return idEM

  def insertResults(self, nameDataset,  idExperiment, fpr, tpr, auc, dataInfo ):
    """

    :param self:
    :param nameDataset:
    :param idExperiment:
    :param fpr:
    :param tpr:
    :param auc:
    :param params:
    :return:
    """


    cur = self.cursor

    blobFPR = cur.var(cx_Oracle.BLOB)
    blobTPR = cur.var(cx_Oracle.BLOB)

    blobFPR.setvalue(0, fpr.tobytes())
    blobTPR.setvalue(0, tpr.tobytes())

    if np.isnan(auc): auc = 0

    self.cursor.callproc("GABD.insertResultats",  [nameDataset, idExperiment, dataInfo["type"], auc, blobFPR, blobTPR])



  def startSession(self):
    self.connectDB
    self._noConnexion__isStarted = True
    return True