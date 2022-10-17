#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:58:39 2022

@author: lawray


-update compareConvergence to allow deal separately with 1st row of training data

"""

import matplotlib.pyplot as plt
import pickle
import numpy as np

class TrainingData():
    """Wrapper that unpacks and stores training data. Conatains:
    
    self.X  -- universal model paremeters (list of 2D numpy arrays)
    self.y  -- posVect 
    self.numSinglets -- number of nonzero terms in the first row of self.X 2D numpy arrays
    
    self.scrambleData() can create randomly ordered (scrambled) training data
    self.X_s
    self.y_s
    
    TO DO:
        -
    """
    
    
    def __init__(self, termHist,vectHist):
        
        self.y = vectHist
        
        # Flatten the termHist into a list of 2D numpy 
        
        # 1. identify the needed dimensionality and the location of single number terms:
        singlet_terms = []
        curve_terms = []
        curve_terms_len = []
        maxDim = 1
        for termPl in range(len(termHist[1])):
            theLen = len(termHist[1][termPl])
            if theLen == 1:
                singlet_terms += [termPl]
            else:
                curve_terms += [termPl]
                curve_terms_len += [theLen]
                if theLen > maxDim:
                    maxDim = theLen
            
        if len(singlet_terms) > maxDim:
            maxDim = len(singlet_terms)
        
        self.numSinglets = len(singlet_terms)
        
        # Now create the empty matrices and populate them
        
        numCurves = 1 + len(curve_terms) # first 'curve' contains the single atom terms
        self.X = []
        for stepPl in range(len(termHist)):
            self.X += [np.zeros((numCurves,maxDim))]
            
            # now loop through the single atom terms
            for sa_pl in range(len(singlet_terms)):
                self.X[stepPl][0,sa_pl] = termHist[stepPl][singlet_terms[sa_pl]][0,0]
                
            # now loop through the curves:
            for curvePl in range(len(curve_terms)):
                self.X[stepPl][curvePl+1,0:curve_terms_len[curvePl]] = termHist[stepPl][curve_terms[curvePl]][:,0]
    
    def scrambleData(self):
        newOrder = random.sample(list(np.arange(len(self.X))), len(self.X))
        
        self.X_s = []
        self.y_s = []

        for theInd in newOrder:
            self.X_s += self.X[theInd]
            self.y_s += self.y_s[theInd]


class MonteCarloOutput():
    
    def __init__(self, energyList=[], goodMoves=[], mc_path=[], trackedEnergyHist=[]):
        if len(energyList) == 0:
            pass
        else:
            self.energyList=energyList
            self.goodMoves = goodMoves
            self.termHist = mc_path[0]
            self.vectHist = mc_path[1]
            self.trackedEnergyHist = trackedEnergyHist
            
            self.goodMoveInds = np.where(np.asarray(goodMoves)==1)[0]
    
    def showEnergiesGood(self, islog=False):
        for pl in range(self.trackedEnergyHist.shape[1]):
            if pl == 0:
                labelName = 'Primary data'
            else:
                labelName= 'Reference set #' + str(pl)
            
            if not islog:
                plt.plot(self.trackedEnergyHist[self.goodMoveInds,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)], label = labelName)
            else:
                plt.plot(np.log(self.trackedEnergyHist[self.goodMoveInds,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)]), label = labelName)
        
        plt.xlabel('Accepted move number')
        plt.ylabel('Quality factor (~total error in Angstroms)')
        plt.legend()
        plt.show()
    
    def showEnergiesAll(self, islog=False):
        for pl in range(self.trackedEnergyHist.shape[1]):
            if pl == 0:
                labelName = 'Primary data'
            else:
                labelName= 'Reference set #' + str(pl)
                
            if not islog:
                plt.plot(np.arange(self.trackedEnergyHist.shape[0]), self.trackedEnergyHist[:,pl], label = labelName)
            else:
                plt.plot(np.arange(self.trackedEnergyHist.shape[0]), np.log(self.trackedEnergyHist[:,pl]), label = labelName)

        plt.xlabel('Attempted move number')
        plt.ylabel('Quality factor (~total error in Angstroms)')
        plt.legend()
        plt.show()
        
    def makeTrainingData(self):
        self.trainingData = TrainingData(self.termHist,self.vectHist)
        
    def compareConvergence(self,otherTrainingData, pl1 = -1, pl2 = -1, useGood=True):
        # ***update to deal separately with the singlet terms (allow scalar shift DOF)
        
        # if hasattr(otherTrainingData, 'trainingData'):
            # if this is a full MC data set, then use the last 

        if useGood: # consider only accepted moves
            pl1 = self.goodMoveInds[pl1]   
            pl2 = otherTrainingData.goodMoveInds[pl2]
        normFact = np.sum(self.trainingData.X[pl1] * self.trainingData.X[pl1])
        normFact *= np.sum(otherTrainingData.trainingData.X[pl2] * otherTrainingData.trainingData.X[pl2])
        normFact = normFact**0.5
            
        # else:
        #     normFact = np.sum(self.trainingData.X[-1]*self.trainingData.X[-1])
        #     normFact *= np.sum(otherTrainingData*otherTrainingData)
        #     normFact = normFact**0.5
            
        return np.sum(self.trainingData.X[pl1] * otherTrainingData.trainingData.X[pl2])/normFact
            
    def load(self, fileName='Training Data.p'):
        with open(fileName, "rb") as f:
            tmp_dict = pickle.load(f)
        
        self.__dict__.clear()
        self.__dict__.update(tmp_dict) 


    def save(self, fileName='Training Data.p'):
        with open(fileName, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
        