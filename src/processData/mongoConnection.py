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





class mongoConnexion(object):
  """

  """

  def __init__(self, cfg):
    '''
        Constructor
        '''

    self.cfg = cfg

  @property
  def connectDB(self):
    """
      Connect to a oracle server given the connexion information saved on the cfg member variable.

      :return: None
    """

    numServers = len(self.cfg.hosts)

    if numServers == 1:
      host = self.cfg.hosts[0]

      if self.cfg.password == "":
          self.cfg.password  = raw_input("Password de l'usuari {} de MongoDB: ".format(self.cfg.username))


      if "ssh" in host:
        sshParams = host["ssh"]

        #DSN =  "{}/{}@localhost:{}/{}".format(self.cfg.username,self.cfg.password,sshParams["port"],self.cfg.sid)
        DSN = "mongodb://{}:{}@{}:{}/{}".format(self.cfg.username, self.cfg.password, host["hostname"], host["port"], self.cfg.db)

        sshParams["password"] = raw_input("Password de l'usuari {} a {}: ".format(sshParams["username"],host["hostname"]))

        self.server = SSHTunnelForwarder((sshParams["hostname"], int(sshParams["port"])),
                                    ssh_username=sshParams["username"],
                                    ssh_password=sshParams["password"],
                                    remote_bind_address=(host["hostname"], host["port"]),
                                    local_bind_address=("", int(sshParams["port"]))
                                    )
        self.server.start()
      else:
        DSN = "mongodb://{}:{}@{}:{}/{}".format(self.cfg.username, self.cfg.password, host["hostname"], host["port"], self.cfg.db)

      self.conn = MongoClient(DSN)
      self.db = self.conn[self.cfg.db]

    else:
      print "Only one server connexion is allowed. Check the config file and run the script again."
      exit(-1)

    return self.conn

  def close(self):
    self.conn.close()
    self.server.stop()

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
      collectionName="nom_de_la_colleccio"

      collection = self.db[collectionName]

      """
      filtre per a la query
      """
      """
      filter_query = {
      "name" : dataset
      }

      res = collection.find(filter_query)
      
      """

      #TODO
      # Comprovar si hi ha cap resultat
      return False


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
        #TODO: insereu les dades
        feature=row.feature



  def insertImageDataset(self, dataset, fileName, params, labels='anno', imageExt='.jpg', labelsExt='.txt'):
    """
      Insert name of images and its labels into the DB.

      :param self:
      :param dataset:
      :param fileName:
      :param params:
      :param labels:
      :param imageExt:
      :param labelsExt:
      :return: True if data has correctly been inserted
    """

    #TODO: Haureu d'implementar aquesta funció més endevant
    #

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

    print "Inserted info image dataset {!r}".format(dataset)
    return True


  def insertDescriptors(self, dataset, features, fileName, params, featuresDir='features'):
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
          print "Warning {} not seems to be a feature file".format(fileName)
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

          #TODO: inseriu les dades en la collecció


      print """{} features of scheme {} correctly inserted """.format(layer, featureSchema)


  def commit(self):
    pass

