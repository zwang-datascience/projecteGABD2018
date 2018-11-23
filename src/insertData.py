# -*- coding: utf-8 -*-
u'''
 
This script insert data into a scheme previously created for outlier detection experiments. Currently, only two DBMS are supported: Oracle and MongoDB.
 
Arguments:
    -c, --config: JSON file with the information required to insert data
    -N, --datasetName: name of the imported dataset 
    -D, --dbms: Database management system used to import data (Oracle or MongoDB)
    -f, --fileName: file where data is stored
 
Created on 3/7/2018
  
@author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
@Institution: Computer Vision Center - Universitat Autònoma de Barcelona
'''

from collections import namedtuple
import getopt
import glob
import json
import os
from os.path import expanduser
import re
import struct
import sys


from processData import config as cfg, mongoConnection as mg, oracleConnection as orcl




# def obtainImageIds(  cur, classList , dataset='mirFlickr' ):
#
#     classes = classList.split(":")
#
#     numClasses = len( classes )
#
#     ids = []
#     if numClasses == 1:
#         query = """select a.image_id
#                 from {}classes a, (select image_id
#                               from {}classes
#                                 group by image_id
#                                 having count(*)=1) b, {} c
#                 where a.image_id=b.image_id and a.image_id=c.image_id and
#                 partition='Train' and a.classe='{}'""".format(dataset,dataset,dataset,classes[0])
#
#         id_list = cur.execute(query)
#                 # Hem de fer les queries per a cada id de la lista
#         ids = [ int(x[0]) for x in id_list ]
#
#     elif numClasses > 1:
#         query = """select m.image_id
#             from {} m, (select image_id
#                             from {}classes
#                             group by image_id
#                             having count(*)={}) t
#             where m.image_id=t.image_id and partition='Train' and
#                 m.image_id not in(select image_id
#                             from (select unique classe
#                                   from {}classes
#                                   where classe in ('{}')) s, (select image_id
#                                                             from {}
#                                                             where partition='Train' ) r
#                             where (r.image_id,s.classe) not in (select r1.image_id,r1.classe
#                                                                 from classes r1))
#             order by 1""".format(dataset,dataset,numClasses,dataset,"','".join(classes),dataset)
#
#         id_list = cur.execute(query)
#                 # Hem de fer les queries per a cada id de la lista
#         ids = [ x[0] for x in id_list ]
#
#     return ids





def insert(dbms, datasetName, fileName, params):
    """
    Insert the data set provided in the file called fileName

    Input arguments:
    :param dbms: object with the data connection
    :param datasetName: Name of the dataset
    :param fileName: full path and name of the file containing the data to be inserted
    :param params: dataset specific parameters required to properly insert the data. This parameter is initialized on tghe config file
    :return: None
    """

    # hem de processar el nom del fitxer de dades per saber a quin dataset pertany
    dataset, b = datasetName.split("/")

    #conn = dbms.connectDB

    if dataset.lower() == "uci":
        collection = b.lower()
        typeDataset = "vector"
        if dbms.exists(collection):
            print("Dataset {} already inserted".format(collection))
            return


        # Afegim la info en la taula general del dataset
        dbms.insertVectorDataset(collection, fileName, params)

    if dataset.lower() == "mirflickr":
        cnn = b.lower()
        typeDataset = "image"
        isInserted = False

        if dbms.exists(dataset.lower()):
            print("Image dataset {} already inserted".format(dataset.lower()))
            isInserted = True
        else:
          """
          Inserim la informació relacionada amb la base de dades de la mirflickr (nom de les imatges i anotacions a 
          la BD
          """
          isInserted = dbms.insertImageDataset(dataset.lower(), fileName, params)

        """
        Inserim els els vectors de característiques associats a la xarxa
        """
        if isInserted:
            dbms.insertDescriptors(dataset, cnn, fileName, params)

    dbms.commit()

    dbms.close()

if __name__ == '__main__':
  # read commandline arguments, first
  fullCmdArguments = sys.argv

  DBMS = ""
  featuresFile = ""
  createDataset = False
  loadFeatures = False
  isConfigFile = False
  unixOptions = "hc:N:D:f:v"
  gnuOptions = ["help", "config_file=", "datasetName=", "dbms=", "fileName=", "verbose"]

  try:
    arguments, values = getopt.getopt(fullCmdArguments[1:], unixOptions, gnuOptions)
  except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

  # evaluate given options
  for currentArgument, currentValue in arguments:
    if currentArgument in ("-v", "--verbose"):
      print ("enabling verbose mode")
    elif currentArgument in ("-h", "--help"):
      print (__doc__)
      sys.exit(0)
      # print ("displaying help")
    elif currentArgument in ("-N", "--datasetName"):
      datasetName = currentValue
    elif currentArgument in ("-D", "--dbms"):
      DBMS = currentValue
    elif currentArgument in ("-c", "--config_file"):
      configfile = currentValue
      isConfigFile = True
    elif currentArgument in ("-f", "--fileName"):
      fileName = currentValue
      loadFeatures = True

  if isConfigFile:
    with open(configfile) as f:
      data = json.load(f)

  if DBMS.lower() == "oracle":
    cfg = cfg.config(data["dbms"][DBMS.lower()])
    db = orcl.oracleConnexion(cfg)

  if DBMS.lower() == "mongodb": db = mg.mongoConnexion(cfg.config(data["dbms"][DBMS.lower()]))

  dataset, _ = datasetName.split("/")
  res = insert(db, datasetName, fileName, data[dataset])

  print("Dades carregades correctament")
  sys.exit(0)
