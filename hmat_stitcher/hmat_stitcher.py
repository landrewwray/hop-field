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
        self.electronNums=self.getElNums(theConfigsWrapper)  #***
        
        
        self.MElists=[]
        for configPl in range(len(theConfigsWrapper.theConfigs)): # ***fix the list name!
            self.MElists += self.makeMElist(theConfigsWrapper,configPl)
            
            
    def makeMElist(self,theConfigsWrapper,theConfigNum):
        """ Creates the hamiltonian lists for each configuration
        """
        #note that config is a set of 5 slightly different coordinate sets
        
        # 1. first define the initial Hamiltonian index for each [atom][orbital] (list of lists of integers)
        #    -- this needs a separate function -- give 's'-orbitals 1 index, 'p': 3 indices
        #    -- reference "theUM.orbSyms[elementNumber]" for the orbital symmetry
        #    -- and number of orbitals, as theUM.orbSyms[elementNumber]==['s',1] or ['p',3]
        
        #***
        
        # 2. next loop through the UM terms (theUM.termsList) and identify a list of pairs for each
        # (this is addressed by the 'ConfigTerms.findPairs' function below, 
        # but one needs the specific maxDist for each UM term)
        
        #***
        
        # 3. Now create the terms, saving the UM call parameters for each ME.
        #    This means first identifying the kind of term (single atom, 2-body interaction, hopping),
        #    and then calling the appropriate function.
        
        # ***To start with, let's just create the single atom terms (on diagonal)
        # Definitions are in UM.Hterm, but the key question is:
        # if theUM.termsList[theTermNum].element1==None --> it's a single-atom term
        # otherwise check theUM.termsList[theTermNum].hop (True/False) to see if it's a hopping term
        
        #*** the returned list will look like matsList[distortionNum][distortionTypeNum][HmatListNum] = np.asarray[[sOrbInd, sOrbInd, E_s]]
        
    def findPairs(self, maxDist):
        #this function is coppied from my old code

        #***
        
        self.pairs_list=[]
        for ii in range(len(self.atoms_list)):
            for jj in range(ii+1,len(self.atoms_list)):
                theDist=np.linalg.norm(self.coords_arry[jj,:]-self.coords_arry[ii,:])
                if theDist <= maxDist:
                    self.pairs_list+=[[ii,jj,theDist]]    
    
    def getElNums(self, theConfigsWrapper):
        """Uses the universal model and configs to generate a list of electron number for each configuration.
        This will reference self.UM.electronsPerAtom and self.UM.elementList
        
        Returns a list of electron number for each configuration.
        """
        #***
        
        
    def save(self,fileName):
        pass

    def makeHmats(self, theUM):
        pass
    
    
    
    
    
    
