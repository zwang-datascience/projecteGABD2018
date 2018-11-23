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

import noConnection as nc

import numpy as np
import utils as u


class mongoConnexion(nc.noConnexion):
  """

  """

  def __init__(self, cfg):
    '''
        Constructor
        '''

    super(mongoConnexion, self).__init__(cfg)


  @property
  def bd(self):
     return super(mongoConnexion, self).bd

  @bd.setter
  def bd(self, nameBD):
    if not self.__isStarted:
      self.connectDB

    self._noConnexion__bd = self._noConnexion__conn[nameBD]


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
        cfg.password  = raw_input("Password de l'usuari {} de MongoDB: ".format(cfg.username))


      if "ssh" in host:
        sshParams = host["ssh"]

        #DSN =  "{}/{}@localhost:{}/{}".format(self.cfg.username,self.cfg.password,sshParams["port"],self.cfg.sid)
        DSN = "mongodb://{}:{}@localhost:{}/{}".format(cfg.username, cfg.password,  sshParams["port"], cfg.db)

        sshParams["password"] = raw_input("Password de l'usuari {} a {}: ".format(sshParams["username"],sshParams["hostname"]))

        self.server = SSHTunnelForwarder((sshParams["hostname"], int(sshParams["port"])),
                                    ssh_username=sshParams["username"],
                                    ssh_password=sshParams["password"],
                                    remote_bind_address=(host["hostname"], host["port"]),
                                    local_bind_address=("", int(sshParams["port"]))
                                    )
        self.server.start()
      else:
        DSN = "mongodb://{}:{}@{}:{}/{}".format(cfg.username, cfg.password, host["hostname"], host["port"], cfg.db)

      self._noConnexion__conn = MongoClient(DSN)
      self._noConnexion__bd = self._noConnexion__conn[cfg.db]

    else:
      print( "Only one server connexion is allowed. Check the config file and run the script again.")
      exit(-1)

    return self._noConnexion__conn

  def close(self):
    self._noConnexion__conn.close()
    if hasattr(self,'server'): self.server.stop()

  def exists(self, dataset):
      """

      :param dataset: name of the dataset
      :return: True if all the feature vectors of the dataset are already inserted in the DB
      """
      # TODO:
      # + Feu la consulta que us permeti determinar si ja heu inserit abans les dades o no. Podeu utilitzar el codi
      # d'exemple que hi ha a continuació

      """
      Poseu el nom de la col·lecció on guardeu la informació dels datasets
      """
      collectionName="Datasets"

      collection = self.db[collectionName]

      """
      filtre per a la query
      """

      filter_query = {
        "name": dataset
      }

      res = collection.find(filter_query)

      # TODO
      # Comprovar si hi ha cap resultat
      return res.count() > 0


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
        #TODO: Haureu de guardar aquesta informació a la BD
        print("k: {}".format(params["k"]))
        k=params["k"]
    else:
        #TODO: Haureu de guardar la informació de la BD sense la inforamció de la K (numero de classes)
        print("Haureu de guardar la informació de la BD sense la inforamció de la K (numero de classes)")



    """
    Anem a llegir el contingut dels fitxers i els guardem en una estructura definida per fila amb camps: id, label, feature
    """
    fila = namedtuple("fila", "id label features ")
    f = open(fileName)
    line = f.readline().rstrip("\n").rstrip("\r")

    features = []
    id = 0
    labels = []
    while line:
      stringFeatures = line.split(",")
      label = stringFeatures[label_pos]
      stringFeatures.remove(label)
      label = label.rstrip('\n')
      feature = [float(x) for x in stringFeatures]
      if len(feature) > 0:
        features.append(fila(id=id, label=label, features=feature))
        id = id + 1

      line = f.readline().rstrip("\n").rstrip("\r")
    f.close()

    """
    Caldra guardar la informació a la BD segons l'estructura que hagueu decidit
    """
    collection = self.bd["Mostres"]
    for row in features:
        #TODO: insereu les dades
        collection.update({"Dataset":nameDataset, "id":row.id},{"Dataset":nameDataset, "id":row.id,"features":row.features, "label":row.label}, upsert=True)

    self.bd["Datasets"].insert_one({"name":nameDataset,"total":id,"type":"vector"})


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

    """
    El següent bucle recorre la carpeta de imatges i extreu la id de la imatge a partir del nom
    """
    collection = self.db["Mostres"]
    listImages = glob.glob(dirImages + '/*' + imageExt)
    for imageName in listImages:
        nom, _ = os.path.splitext(os.path.basename(imageName))
        match = re.search(r'im(?P<id>\d+)', nom)

        data = {'name': nom, 'image_id': int(match.group('id'))}

        # TODO: caldra inserir la informació en la BD
        collection.update({"Dataset": dataset, "id": data["image_id"]},
                      {"Dataset": dataset, "id": data["image_id"], "name": data["name"]}, upsert=True)

    self.db["Datasets"].insert_one({"name": dataset, "total": len(listImages)})
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
                    res = collection.find({"Dataset": dataset, "id": data["image_id"]})
                    if res.count() > 0:
                      for row in res:
                        if "label" in row:
                          label = row["label"] + [nom]
                        else:
                          label = [nom]
                        collection.update({"Dataset": dataset, "id": row["id"]},{"Dataset": dataset, "id": row["id"], "name": row["name"], "label": label}, upsert=True)
                    #else:
                    #  label = [nom]

                    line = f.readline().rstrip("\n").rstrip("\r")

            f.close()

    #self.commit()

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
      El següent codi llegeix els fitxers de caracteristiques i els processa. 
      """

      dataDir = "/".join(os.path.dirname(fileName).split("/")[0:-1])
      dirImages = dataDir + "/" + featuresDir


      featureSchema, fileExt = os.path.splitext(fileName)
      if fileExt[-8:].lower() != 'features':
          print ("Warning {} not seems to be a feature file".format(fileName))
      layer = fileExt[1:-8]

      # load feature vector
      f = open(fileName, 'r')
      dims = [int(val) for val in f.readline().rstrip("\n").split(" ")]
      x = f.read()
      f.close()
      size = int(params['feat_size'])  # bytes to represent each float
      type = params['type']

      collection = self.bd["Mostres"]
      for i in range(0, dims[0]):
          # cur.execute(None, {'image_id': i, 'layer': layer, 'features': Features[i] })
          bf = x[i * dims[1] * size:(i + 1) * size * dims[1]]
          numFeatures = len(bf) / size
          features = list(struct.unpack(type * numFeatures, bf))

          res = collection.find({"Dataset": dataset.lower(), "id": i + 1,"cnn": { "$exists" : "true"}})
          if res.count() > 0:
              record = collection.find({"Dataset": dataset.lower(), "id": i + 1,
                                        "cnn": {"$elemMatch": {"name": featuresName, "layer": layer}}})
              if record.count() == 0:
                  collection.update({"Dataset": dataset.lower(), "id": i + 1},
                            {"$push" : {"cnn":  {"name": featuresName, "layer": layer, "features": features } } } )
          else:
              collection.update({"Dataset": dataset.lower(), "id": i + 1},
                              {"$set": { "cnn": [ {"name": featuresName, "layer": layer, "features": features } ] }})


      print ("""{} features of scheme {} correctly inserted """.format(layer, featureSchema))

  def insertExperiment(self, conf, repeticio,  method, paramsMethod):
    '''
    Inserim en la taula Experiment la informació bàsica de l'experiment, si no existeix. Si existeix retorna el OID del document

    :param conf: ratio of class outliers and attribute outliers
    :param method: name of the evaluated outlier detection algorithm
    :param paramsMethod: list of parameter names
    :return: idExperiment


    '''


    """inserim la informació dels experiments"""
    collection = self.bd["Experiments"]

    #TODO: SESSIO 15 Busca a la col·leccio si el document hi és.
    res = []

    if res.count()>0:
      idEM = res[0]["_id"]
    else:
      #TODO: SESSIO 15 Si no hi és, insereu-lo
      res = []
      idEM = res

    return idEM


  def insertOutlierData(self, newFeatures, nameDataset, repeticio, outliersGTIdx, conf , dataInfo ):
    """
    Inserim els outliers

    :param newFeatures:
    :param nameDataset:
    :param repeticio:
    :param outliersGTIdx:
    :param conf:
    :param dataInfo:
    :return:
    """


    numViews = len(newFeatures)

    collection = self.bd["Outliers"]

    for i in outliersGTIdx:
      # Hem de construir el vector de caracteristiques complet
      features = np.hstack([newFeatures[y][i].features for y in range(numViews)]).tolist()




      if dataInfo["type"] == "vector":
        print("TODO:")
        #TODO: SESSIÓ 15: Insereu/actualitzeu els outliers pels datasets de Vectors (UCI)

      if dataInfo["type"] == "image":
        print("TODO:")
        #TODO: SESSIÓ 15: Insereu/actualitzeu els outliers pels datasets d'imatges  (MIRFLICKR)



    pass




  def insertResults(self, nameDataset, idExperiment, fpr, tpr, auc, dataInfo):
    """
    inserir els resultats

    :param nameDataset:
    :param idExperiment:
    :param fpr:
    :param tpr:
    :param auc:
    :param dataInfo:
    :return:
    """

    collection = self.bd["Results"]

    if dataInfo["type"] == "vector":
      print("TODO:")
      # TODO: SESSIÓ 15: Insereu/actualitzeu els Resultats  pels datasets de Vectors (UCI)

    if dataInfo["type"] == "image":
      print("TODO:")
      # TODO: SESSIÓ 15: Insereu/actualitzeu els Resultats pels datasets d'imatges  (MIRFLICKR)



  def loadOutliers(self, nameDataset, repeticio, numSamples, conf, dataInfo):
    """
    Cal llegir els outliers

    :param nameDataset:
    :param repeticio:
    :param numSamples:
    :param conf:
    :param dataInfo:
    :return:
    """



    numTotalOutliers = int(2 * round(conf[0] / 100.0 / 2.0 * numSamples)) + int(round(conf[1] / 100.0  * numSamples))

    collection = self.bd["Outliers"]


    if dataInfo["type"] == "vector":
      res = []
      # TODO: SESSIÓ 15: carregueu els outliers dels datasets de Vectors (UCI)

    if dataInfo["type"] == "image":
      res = []
      # TODO: SESSIÓ 15: Icarregueu els outliers dels datasets d'imatges  (MIRFLICKR)

    numOutliers =res.count()
    generateOutliersFlag = numTotalOutliers != numOutliers
    outliers = {}

    if not generateOutliersFlag:
      for row in res:
        outliers[row["id"]] = row["features"]

    return outliers,generateOutliersFlag

  def __loadVectorData(self, nameDataset, dataInfo):
    '''
    Carrega els vectors de caracteristiques dels datasets de la UCI

    :param nameDataset:
    :param dataInfo:
    :return:
    '''


    collection = self.bd["Mostres"]

    res = collection.find({"Dataset": nameDataset},{"id":1,"features":1,"label":1})


    fila = namedtuple("fila", "id features")


    taula = []
    ids = {}
    for row in res:
        taula.append(fila(id=row["id"], features=row["features"] ))
        if row["label"] in ids.keys():
            ids[row["label"]] = ids[row["label"]] + [row["id"]]
        else:
            ids[row["label"]] = [row["id"]]

    return taula, ids

  def __loadImageData(self, nameDataset, dataInfo):
    '''
    Carrega els vectors de caracteristiques dels datasets de la UCI

    :param nameDataset:
    :param dataInfo:
    :return:
    '''


    collection = self.bd["Mostres"]

    res = collection.find({"Dataset": nameDataset, "cnn": dataInfo["cnn"], "layer": dataInfo["layers"]},{"id":1,"features":1,"label":1})


    fila = namedtuple("fila", "id features")


    taula = []
    ids = {}
    for row in res:
        taula.append(fila(id=row["id"], features=row["features"] ))
        if row["label"] in ids.keys():
            ids[row["label"]] = ids[row["label"]] + [row["id"]]
        else:
            ids[row["label"]] = [row["id"]]

    return taula, ids

  def loadData(self,nameDataset, data):

    if data.type == "vector":
      features, classIds = self.__loadVectorData(nameDataset, data)

    if data.type == "image":
        features, classIds = self.__loadImageData(nameDataset, data)

    return features, classIds

  def getDatasetType(self,nameDataset):
    """

    :param nameDataset:
    :return:
    """

    collection = self.bd["Datasets"]

    res = collection.find({"name": nameDataset.lower()},{"type":1})

    if res.count() > 0:
      return res[0]["type"]
    else:
      return "unknown"


  def startSession(self):

    self.connectDB
    self._noConnexion__isStarted = True
    return True



