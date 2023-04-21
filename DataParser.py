# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 09:20:08 2016

@author: matevzk
"""

import DataBin as db
from sklearn.model_selection import KFold as KF
import numpy
import logging
import bralec as cnf



def createTrainTestSet():
    datasetName = cnf.GE_cf.get('dataSet','dataName')
    fold = cnf.GE_cf.getboolean("datasetFolding","fold")
    
    db.data_set=numpy.loadtxt(datasetName, delimiter=';')
    if fold:
        foldNum = cnf.GE_cf.getint("datasetFolding","numOfFolds")
        db.fold_bin = KF(n_splits=foldNum,random_state=None, shuffle=False)
    else:    
        numpy.random.shuffle(db.data_set)
        split = round(db.data_set.shape[0]*0.8)
        split = int(split)
        training, test = db.data_set[:split,:], db.data_set[split:,:]
        db.trainSet = training
        db.testSet = test
        db.calculateStaticBiases()
        logging.debug( "datasetSize" + str( db.trainSet.shape[0]))
    
    
def setFold(n): 
    i = 1
    for train_index, test_index in db.fold_bin.split(db.data_set):
        if i==n:
            db.trainSet, db.testSet = db.data_set[train_index], db.data_set[test_index]
            db.resetMatrices()
            db.calculateStaticBiases()

        i = i + 1
        
    logging.debug( "datasetSize" + str( db.trainSet.shape[0]))
    
