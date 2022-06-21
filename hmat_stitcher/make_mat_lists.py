#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022



@author: lawray
"""


class ConfigTerms:
    def __init__(self, theUM, theConfigsWrapper):
        """
        theUM is a UniversalModel.  For this init, the Hamiltonian symmetries
        will matter, but the parameters multiplied onto them will not
        
        theConfigsWrapper is a list of atomic configurations, elements, etc
        """
        
        self.UM = theUM  # to remember they types of Hamiltonian terms
        self.electronNums=self.getElNums()
        
        
        self.MElists=[]
        for pl in range(theConfigs):
            self.MElists += self.makeMElist(theConfigs[pl],theElements[pl])
            
            
    def makeMElist(self,theConfigsWrapper,theConfigNum):
        #note that config is a set of 5 slightly different coordinate sets
        
        # 1. first define Hamiltonian indices -- this needs a separate function
        
        # 2. next loop through the UM terms and identify a list of pairs for each
        # (this is )
        
        
        # 3. now create the terms, saving the UM call parameters for each ME
        
    def getElNums(self):
        """Uses the universal model and configs to generate a list of electron number for each configuration.
        This will reference UM.electronsPerAtom and UM.elementList
        
        """
        
    def save(self,fileName):
        pass

    def makeHmats(self, theUM):
        pass
    
    
    
    
    
    
