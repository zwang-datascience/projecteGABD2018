# -*- coding: utf-8 -*-
u'''
 
This script evaluates the performance of the following outlier detection method:
    - Consensus Regularized Multi-View Outlier Detection (CMOD)
    - DMOD
    - HOAD

Arguments:
    -c, --config: JSON file with the information required to insert data
    -N, --datasetName: name of the imported dataset
    -D, --dbms: Database management system used to import data (Oracle or MongoDB).
    -f, --featuresImage: extracted features from image dataset. e.g -f "{'cnn':'AlexNet', 'layer':'Visual'}"
    -m, --method: coma-separated list with the outlier detection methods to test (either CMOD, DMOD or HOAD)
    -p, --params: string on JSON format with the method parameters and their values. e.g. -p "{'k':2, 'sigma':.1, 'm':1}"

Created on 26/2/2018
  
@author: Oriol Ramos Terrades (oriolrt@cvc.uab.cat)
@Institution: Computer Vision Center - Universitat Autonoma de Barcelona
'''


__author__ = 'Oriol Ramos Terrades'
__email__ = 'oriolrt@cvc.uab.cat'





import ast
import getopt
import glob
import json
import numpy as np
import os.path

import sys

from pprint import pprint
from collections import namedtuple


from scipy.spatial import distance as dist
from sklearn import metrics

from OutlierDetector.CMOD import CMOD
from OutlierDetector.DMOD import DMOD
from OutlierDetector.HOAD import HOAD

from processData import config as cfg, mongoConnection as mg, oracleConnection as orcl, noConnection as nc
from processData.datasetInfo import datasetInfo as dsi



def getConf( confDict ):

    confString = confDict.split(" ")
    conf = []
    for c in confString:
        conf.append(tuple([int(x) for x in c.replace("(", "").replace(")", "").split(",")]))


    return conf



def loadData(dbms, nameDataset, params={}):
    '''
    Description: load data from the DBMS given the parameters passed by argument

    :param dbms: object with the connection to the DB
    :param nameDataset:
    :param params:
    :return: list of rows with the id of the image and the feature vector as numpy array
    '''

    data = dsi(nameDataset)

    if "classList" in params:
      data.classList = params["classList"]
    else:
      data.classList = []

    if "layers" in params:
        data.layers = params["layers"]
    else:
        data.layers = ["Visual"]

    if "cnn" in params:
        data.layers = params["cnn"]
    else:
        data.layers = ""


    data.type = dbms.getDatasetType(nameDataset)
    data.features, data.classIds = dbms.loadData(nameDataset, data )


    return data


if __name__ == '__main__':
    # read commandline arguments, first
    fullCmdArguments = sys.argv

    #classPath = "jars"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #jarFiles = glob.glob(dir_path + '/' + classPath + '/' + '*.jar')

    """
    Database default parameters
    """
    DBMS = ""
    methods = ["DMOD", "CMOD", "HOAD"]

    numViews = 2
    params = {}
    datasetName = "Synthetic Data"
    params["numSamples"] = 200
    isConfigFile = False

    unixOptions = "hvc:f:N:m:D:p:"
    gnuOptions = ["help", "verbose", "config_file=", "datasetName", "featuresImage=","method=", "dbms=", "params="]

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
        elif currentArgument in ("-c", "--config_file"):
            configFile = currentValue
            isConfigFile = True
        elif currentArgument in ("-D", "--dbms"):
            DBMS = currentValue
        elif currentArgument in ("-m", "--method"):
            method = currentValue
        elif currentArgument in ("-N", "--datasetName"):
            datasetName = currentValue.lower()
        elif currentArgument in ("-p", "--params"):
            paramsMethod = ast.literal_eval(currentValue)
        elif currentArgument in ("-f", "--featuresImage"):
            featuresImage = ast.literal_eval(currentValue)

    if isConfigFile:
        with open(configFile) as f:
            data = json.load(f)

    if DBMS.lower() == "":
        db = nc.noConnexion()

    if DBMS.lower() == "oracle":
        db = orcl.oracleConnexion(cfg.config(data["dbms"][DBMS.lower()]))

    if DBMS.lower() == "mongodb":
        db = mg.mongoConnexion(cfg.config(data["dbms"][DBMS.lower()]))

    """Iniciem la sessio"""
    db.startSession()

    """Carreguem les dades dels datasets guardats a la BD"""
    dataInfo = loadData(db, datasetName, params)
    """---"""

    paramNames = []
    if "data" in locals():
        if "numIterations" in data:
            numRepeticions = int(data['numIterations'])
        else:
            numRepeticions = 2
        if "conf" in data:
            confList = getConf(data['conf'])
        else:
            confList = [(2,0)]
    else:
        numRepeticions = 2
        confList = [(2,0)]

    for conf in confList:
        """Inicialitzem"""
        if method.upper() == "DMOD":
            od = DMOD(numViews)
        if method.upper() == "CMOD":
            od = CMOD(numViews)
        if method.upper() == "HOAD":
            od = HOAD(numViews)

        for i in range(numRepeticions):
            """Per a cada repetició hem de generar els outliers del dataset """
            print("""
            ==================================================================
            
            Iniciant repetició {}
            
            """.format(i))

            newFeatures, y, outliersGTIdx = od.prepareExperimentData(db, conf,  datasetName, dataInfo, i, settings={'numViews':numViews})

            idExperiment = db.insertExperiment(conf, i, method, paramsMethod)

            outliersIdx = od.detector(newFeatures, paramsMethod )

            """Calculem les mètriques d'avaluació"""
            # Evaluate Outliers
            fpr, tpr, thresholds = metrics.roc_curve(y, outliersIdx, pos_label=1)
            auc = metrics.auc(tpr, fpr)

            """Inserim els resultats a la BD """
            db.insertResults(datasetName, idExperiment, fpr, tpr, auc, dataInfo)

            """Mostrem els resultats per consola"""
            valorsStr = "{}: {}".format(dataInfo, method)
            for key in paramsMethod:
                valorsStr = valorsStr + ", {}={}".format(key, paramsMethod[key])
            valorsStr = valorsStr + ", {}-{} (repeticio {}): %.3f".format(conf[0],conf[1],i) %(auc)

            print(valorsStr)

    db.close()
    print("Experiments fets")
    sys.exit(0)