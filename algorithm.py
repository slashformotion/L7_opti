# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 08:33:41 2015

@author: matevzk
"""
import numpy
import logging
import bralec as cnf
import DataBin as db
import DataParser as dp



logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.WARNING)


##########################################################################################              
### INITIALIZATION  
##########################################################################################

class Algorithm(object):
    def __init__(self, code=None):  #sets some of the initial parameters
        print (cnf.alg_cf.getfloat('mfParameters','plearningrate'))
        if code == None:
            self.updateFunction = compile("self.tmpUF = self.tempUF + (error * self.tempIF - self.regularization * self.tempUF) * self.pLearningRate; self.tmpIF = self.tempIF + (error * self.tempUF - self.regularization * self.tempIF) * self.qLearningRate ","<string>","exec")
        else:
            self.updateFunction = code
        self.pLearningRate = cnf.alg_cf.getfloat("mfParameters","plearningrate")
        self.qLearningRate = cnf.alg_cf.getfloat("mfParameters","qlearningrate")
        self.regularization = cnf.alg_cf.getfloat("mfParameters","regularization")
        self.numOfFeatures = cnf.alg_cf.getint("mfParameters","numOffeatures")
        self.numOfIterations = cnf.alg_cf.getint("mfParameters","numofiterations")
        self.initFeatureValue = cnf.alg_cf.getfloat("mfParameters","initfeaturevalue")
        
        self.maxValue = cnf.alg_cf.getfloat("mfParameters","maxValue")

        
        self.userFeaturesStorage = [] 
        self.itemFeaturesStorage = []
        self.userFeaturesMatrix = [] 
        self.itemFeaturesMatrix = []
 
   
##########################################################################################              
### DATA ABSTRACTION 
##########################################################################################
    
    def getUserFeatureVector(self,userID):
        if userID < self.userFeaturesMatrix.shape[0]:
            return self.userFeaturesMatrix[int(userID),0:]
        else:
            return numpy.ones(self.numOfFeatures)*self.initFeatureValue    
    
    def getItemFeatureVector(self,itemID):
        if itemID < self.itemFeaturesMatrix.shape[0]:
            return self.itemFeaturesMatrix[int(itemID),0:]
        else:
            return numpy.ones(self.numOfFeatures+1)*self.initFeatureValue     
   
##########################################################################################              
### RATING CALCULATION  
##########################################################################################
        
    def calculateRating(self,uID,iID):
        minRating = cnf.alg_cf.get('dataInfo', 'minrating')
        maxRating = cnf.alg_cf.get('dataInfo', 'maxrating')
        u= db.globalBiasMatrix
        bi = db.getItemBias(iID)
        bu = db.getUserBias(uID)
        p = self.getUserFeatureVector(uID)
        q = self.getItemFeatureVector(iID)
        r=self.plainModel(u,bu,bi,p,q)
        return self.fixPredictedRating(r, float(minRating), float(maxRating))
        
    def fixPredictedRating (self, predictedR, minRating, maxRating ):
        if predictedR > maxRating:
            predictedR = maxRating
        elif predictedR < minRating:
            predictedR = minRating
        else:
            predictedR = predictedR
        return predictedR 
        
##########################################################################################              
### MODEL  
##########################################################################################   
        
    def plainModel (self, u, bu, bi, p, q ):   
        if p.shape[0] == self.numOfFeatures:
            p = numpy.append(p,0)
        pq = numpy.matrix(p)*numpy.matrix(q).T
        return u + bu + bi + pq
        


##########################################################################################              
### EVALUATION  
##########################################################################################

    def evaluateModel(self,row = 0):
        self.userFeaturesMatrix = self.userFeaturesStorage[row]
        self.itemFeaturesMatrix = self.itemFeaturesStorage[row]
        n = db.testSet.shape[0]
        sumError = 0
        for k in range(db.testSet.shape[0]):
            predRating = self.calculateRating(db.testSet[k,0],db.testSet[k,1])
            trueRating = db.testSet[k,2]
            sumError += (predRating-trueRating)**2
        sumError = sumError / n
        sumError = numpy.sqrt(sumError)
        return float(sumError)
        
    def evaluateFoldedModel(self):
        foldNum = cnf.GE_cf.getint("datasetFolding","numOfFolds")
        total_cost = 0.0
        for i in range(foldNum):
            total_cost += self.evaluateModel(i)
        
        return total_cost / (1.0*foldNum) 
        
    def getProgramCost(self):
        cost = 200
        fold = cnf.GE_cf.getboolean("datasetFolding","fold")
        if fold:
            cost = self.evaluateFoldedModel()
        else:
            cost = self.evaluateModel()
        return cost
 

       
##########################################################################################              
### BASELINE CALCULATION  
##########################################################################################
        
    def getBaseline(self):
        fold = cnf.GE_cf.getboolean("datasetFolding","fold")
        baseline = None
        if fold:
            foldNum = cnf.GE_cf.getint("datasetFolding","numOfFolds")
            # print foldNum
            logging.info("Calculating folded baseline: " + str(foldNum)) 
            baseline =  self.createFoldedBaseline(foldNum)
        else:
            logging.info("Calculating baseline") 
            baseline =  self.createBaseline()
        logging.info("MF algorithm RMSE = " + str( baseline))
        return baseline

    def createBaseline(self, row = 0):    
        maxUsrID = int(numpy.amax(db.trainSet[:,0]))
        maxItmID = int(numpy.amax(db.trainSet[:,1]))
           
        logging.info(" MaxUserId = " + str(maxUsrID))
        logging.info(" MaxItemId = " + str(maxItmID))
          
        pMatrix = numpy.zeros([maxUsrID+1,self.numOfFeatures+1])
        qMatrix = numpy.zeros([maxItmID+1,self.numOfFeatures+1])   
          
        for f in range(self.numOfFeatures):
            
            logging.info("Feature = " + str(f))
            
            pMatrix[:,f]=self.initFeatureValue
            qMatrix[:,f]=self.initFeatureValue
            
            for k in range(self.numOfIterations):
                for i in range(db.trainSet.shape[0]):
                    userID = int(db.trainSet[i,0])
                    itemID = int(db.trainSet[i,1])
                    trueRating = db.trainSet[i,2]
                    estimatedRating = self.plainModel (db.globalBiasMatrix, db.getUserBias(userID), db.getItemBias(itemID), pMatrix[userID,:], qMatrix[itemID,:])
                    error = trueRating - estimatedRating
    
                    self.tempUF = pMatrix[userID,f];
                    self.tempIF = qMatrix[itemID,f];
                    exec(self.updateFunction)
                    pMatrix[userID,f] = numpy.double(self.tmpUF)
                    qMatrix[itemID,f] = numpy.double(self.tmpIF)                        
              
        
        self.userFeaturesStorage.append(pMatrix)
        self.itemFeaturesStorage.append(qMatrix)      
        cost = self.evaluateModel(row)
        return cost


    def createFoldedBaseline(self,n):
        total_cost = 0.0
        for i in range(1,n+1):
            dp.setFold(i)
            total_cost += self.createBaseline((i-1))
        return (total_cost/n)
        
    

 