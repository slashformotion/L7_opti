# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 09:19:57 2016

@author: matevzk
"""
import numpy
#import config.parser as cnf
#from sklearn.cross_validation import KFold as KF

### data stroage
fold_bin = None
data_set = None
trainSet = None
testSet = None

### features and biases
userBiasesMatrix = None
itemBiasesMatrix = None
globalBiasMatrix = None



### Basic features setup and reset (for folding)
def resetMatrices():
    global userBiasesMatrix
    global itemBiasesMatrix
    global globalBiasMatrix
    userBiasesMatrix = None
    itemBiasesMatrix = None
    globalBiasMatrix = None
    
def calculateStaticBiases ():
    global trainSet
    data = trainSet
    global globalBiasMatrix
    global userBiasesMatrix
    global itemBiasesMatrix
    ## calculate global bias
    # calculate mean of all the ratings
    globalBias = numpy.zeros([1,1])
    globalBias[0,0] = numpy.mean(data[:,2])
    globalBiasMatrix = globalBias;
    
    # prepare the users' biases matrix    
    numberOfUsers = len(numpy.unique(data[:,0]))
    userMatIdBias = numpy.zeros([numberOfUsers, 2])
    
    # prepare the items' biases matrix
    numberOfItems = len(numpy.unique(data[:,1]))
    itemMatIdBias = numpy.zeros([numberOfItems, 2])
        
    ## calculate user bias    
    # calculate mean rating for the user in the condition
    for i, usrIndex in enumerate(numpy.unique(data[:,0])):
        condition = data[:,0]==usrIndex
        f=numpy.mean(numpy.extract(condition, data[:,2]))
        f=f-globalBias[0,0]
        userMatIdBias[i,:] = [usrIndex,f]
        
    
    userBiasesMatrix =  userMatIdBias
    ## calculate item bias 
    # calculate mean rating for the item in the condition
    for i, itmIndex in enumerate(numpy.unique(data[:,1])):
        condition = data[:,1]==itmIndex
        f=numpy.mean(numpy.extract(condition, data[:,2]))
        f = f-globalBias[0,0]
        itemMatIdBias[i,:] = [itmIndex,f]

    itemBiasesMatrix = itemMatIdBias
    
    
### Bias retrieval functions    
def getUserBias(userID):
    global userBiasesMatrix
    usrIndexList = list(userBiasesMatrix[:,0])
    index = usrIndexList.index (userID) if userID in usrIndexList else None
    if index != None:
        return userBiasesMatrix[index,1]
    else:
        return 0

def getItemBias(itemID):
    global itemBiasesMatrix
    itmIndexList = list(itemBiasesMatrix[:,0])
    index = itmIndexList.index (itemID) if itemID in itmIndexList else None
    if index != None:
        return itemBiasesMatrix[index,1]
    else:
        return 0