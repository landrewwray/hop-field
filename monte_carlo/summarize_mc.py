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
import random

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
            self.y[stepPl] = np.asarray(self.y[stepPl])
            
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
            # print(theInd)
            # print(self.X[theInd])
            # print(self.y[theInd])
            self.X_s += [self.X[theInd]]
            self.y_s += [self.y[theInd]]


class MonteCarloOutput():
    
    def __init__(self, energyList=[], goodMoves=[], mc_path=[], trackedEnergyHist=[], angle_length_errorHist = [], setNames=[]):
        if len(energyList) == 0:
            pass
        else:
            self.energyList=energyList
            self.goodMoves = goodMoves
            self.termHist = mc_path[0]
            self.vectHist = mc_path[1]
            self.trackedEnergyHist = trackedEnergyHist
            self.angle_length_errorHist = angle_length_errorHist
            self.setNames = setNames
            
            self.goodMoveInds = np.where(np.asarray(goodMoves)==1)[0]
    
    def showBondError(self, stretch1_angle0 = 1, islog=False, showLegend=True):
    
        fig = plt.figure(figsize=(6, 7.5))
        ax = fig.add_subplot(1, 1, 1)     
        
        for pl in range(self.angle_length_errorHist.shape[2]):
            offsetVal = pl * (0.5 + (1-stretch1_angle0) * 49.5) 
            
            if len(self.setNames) > 0:
                labelName = self.setNames[pl]
            elif pl == 0:
                labelName = 'Primary data'
            else:
                labelName= 'Reference set #' + str(pl)
            
            if not islog:
                plt.plot(offsetVal + self.angle_length_errorHist[self.goodMoveInds,stretch1_angle0,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)], label = labelName)
            else:
                plt.plot(offsetVal + np.log(self.angle_length_errorHist[self.goodMoveInds,stretch1_angle0,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)]), label = labelName)
            plt.axhline(y=offsetVal, color='black', linestyle='-')
        
        plt.xlabel('Accepted move number')
        if stretch1_angle0:
            plt.ylabel('Bond length error, mean of abs(Angstroms)')    
        else:
            plt.ylabel('Bond angle error, mean of abs(degrees)')    
            
        # plt.legend()
        if showLegend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Molecules', loc='upper right')
        plt.show()
    
    def showEnergiesGood(self, islog=False, showLegend=True):
        
        fig = plt.figure(figsize=(6, 7.5))
        ax = fig.add_subplot(1, 1, 1)
            
        for pl in range(self.trackedEnergyHist.shape[1]):
            offsetVal = pl*5
            
            
            if len(self.setNames) > 0:
                labelName = self.setNames[pl]
            elif pl == 0:
                labelName = 'Primary data'
            else:
                labelName= 'Reference set #' + str(pl)
            
            if not islog:
                plt.plot(offsetVal + self.trackedEnergyHist[self.goodMoveInds,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)], label = labelName)
            else:
                plt.plot(offsetVal + np.log(self.trackedEnergyHist[self.goodMoveInds,pl+np.zeros(self.goodMoveInds.shape[0],dtype=np.int32)]), label = labelName)
            plt.axhline(y=offsetVal, color='black', linestyle='-')
        
        plt.xlabel('Accepted move number')
        plt.ylabel('Quality factor (~total error in Angstroms)')
        if showLegend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Molecules', loc='upper right')
        plt.show()
    
    def showEnergiesAll(self, islog=False, showLegend=True):
        
        fig = plt.figure(figsize=(6, 7.5))
        ax = fig.add_subplot(1, 1, 1)
        
        for pl in range(self.trackedEnergyHist.shape[1]):
            offsetVal = pl*5
            
            if len(self.setNames) > 0:
                labelName = self.setNames[pl]
            elif pl == 0:
                labelName = 'Primary data'
            else:
                labelName= 'Reference set #' + str(pl)
                
            if not islog:
                plt.plot(np.arange(self.trackedEnergyHist.shape[0]), offsetVal + self.trackedEnergyHist[:,pl], label = labelName)
            else:
                plt.plot(np.arange(self.trackedEnergyHist.shape[0]), offsetVal + np.log(self.trackedEnergyHist[:,pl]), label = labelName)
            plt.axhline(y=offsetVal, color='black', linestyle='-')
            
        plt.xlabel('Attempted move number')
        plt.ylabel('Quality factor (~total error in Angstroms)')
        if showLegend:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], title='Molecules', loc='upper right')
        plt.show()
        
    def makeTrainingData(self):
        self.trainingData = TrainingData(self.termHist,self.vectHist)
        
    def compareConvergence(self,otherTrainingData, pl1 = -1, pl2 = -1, useGood=True):
        """normXcorr comparison of converged model parameters and accuracy vectors.
        
        Normalization eliminates the amplitude degree of freedom, and the orbital
        energies are anchored by setting the hydrogen s-orbital to 0 eV.
        """
        
        # if hasattr(otherTrainingData, 'trainingData'):
            # if this is a full MC data set, then use the last 

        if useGood: # consider only accepted moves
            pl1 = self.goodMoveInds[pl1]   
            pl2 = otherTrainingData.goodMoveInds[pl2]
        normFact = np.sum(self.trainingData.X[pl1] * self.trainingData.X[pl1])
        normFact *= np.sum(otherTrainingData.trainingData.X[pl2] * otherTrainingData.trainingData.X[pl2])
        normFact = normFact**0.5
            
        normFact_y = np.sum(self.trainingData.y[pl1] * self.trainingData.y[pl1])
        normFact_y *= np.sum(otherTrainingData.trainingData.y[pl2] * otherTrainingData.trainingData.y[pl2])
        normFact_y = normFact_y**0.5
        
        
        # else:
        #     normFact = np.sum(self.trainingData.X[-1]*self.trainingData.X[-1])
        #     normFact *= np.sum(otherTrainingData*otherTrainingData)
        #     normFact = normFact**0.5
            
        return [np.sum(self.trainingData.X[pl1] * otherTrainingData.trainingData.X[pl2])/normFact, np.sum(self.trainingData.y[pl1] * otherTrainingData.trainingData.y[pl2])/normFact_y]

    def compare_with_step(self, stepSize, startPl = 0):
        """Look at attempted configs with a fixed step size and compare the last
        'good step' configurations.  The result will be two correlation matrices,
        one for X and one for y.
        """
        
        # first loop through to all eligible configs:
        indList=[]
        numPts=int(np.floor((len(self.trainingData.X)-startPl)/stepSize))
        for thePt in range(numPts):
            #find the last good spot before [startPl+(thePt+1)*stepSize]
            comparePl=startPl+(thePt+1)*stepSize
            while not self.goodMoves[comparePl]:
                comparePl-=1  # Assuming there's a valid move to find!
            
            indList+=[comparePl]
        
        #now create the matrices:
        x_corrMat=np.zeros((len(indList),len(indList)))
        y_corrMat=np.zeros((len(indList),len(indList)))
        for ind1 in range(len(indList)):
            for ind2 in range(len(indList)):
                xComp, yComp = self.compareConvergence(self, pl1=indList[ind1], pl2=indList[ind2], useGood=False)
                x_corrMat[ind1,ind2] = xComp
                y_corrMat[ind1,ind2] = yComp
        
        return x_corrMat, y_corrMat

    def load(self, fileName='Training Data.p'):
        with open(fileName, "rb") as f:
            tmp_dict = pickle.load(f)
        
        self.__dict__.clear()
        self.__dict__.update(tmp_dict) 


    def save(self, fileName='Training Data.p'):
        with open(fileName, "wb") as f:
            pickle.dump(self.__dict__, f, 2)

