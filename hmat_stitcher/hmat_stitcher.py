#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022
@author: lawray
"""

import numpy as np

class ConfigTerms:
    def __init__(self, theUM, theConfigsWrapper):
        """
        theUM is a UniversalModel.  For this init, the Hamiltonian symmetries
        will matter, but the parameters multiplied onto them will not
        
        theConfigsWrapper is a list of atomic configurations, elements, etc
        """
        
        self.UM = theUM  # to remember they types of Hamiltonian terms
        self.electronNums=self.getElNums(theConfigsWrapper)  #***
        
        
        self.MElists=[]   # initialized by makeMElist
        self.pairsList=[] # initialized by makeMElist
        self.hmatIndex=[] # values created inside makeMElist
        self.hmatTermLists=[] # values created inside makeMElist
        for molPl in range(len(theConfigsWrapper.distortLists)): # ***fix the list name!
            self.MElists += self.makeMElist(theConfigsWrapper,molPl)
            
            
    def makeMElist(self,theConfigsWrapper,molPl):
        """ Creates the hamiltonian lists for each configuration
        """
         #note that config is a set of 5 slightly different coordinate sets
        
        # 1. first define the initial Hamiltonian index for each [atom][orbital] (list of lists of integers)
        #    -- this needs a separate function -- give 's'-orbitals 1 index, 'p': 3 indices
        #    -- reference "theUM.orbSyms[elementNumber]" for the orbital symmetry
        #    -- and number of orbitals, as theUM.orbSyms[elementNumber]==['s',1] or ['p',3]
        
        self.hmatIndex+=[self._makeHamiltonianIndex(theConfigsWrapper,molPl)]
        
        # 2. next loop through the UM terms (theUM.termsList) and identify a list of pairs for each
        # (this is addressed by the 'ConfigTerms.findPairs' function below, 
        # but one needs the specific maxDist for each UM term)

        #*** should create this:
        #*** self.pairsList[moleculeNum][distortBondNum][distortionNum_0_thru_4][pairNum][atom_1_or_2_or_dist]
        #  e.g. self.pairsList[3][2][3][17] == [3, 5, 0.784832]
        self.pairsList += [self.findPairs(theConfigsWrapper, self.UM.maxDist, molPl)]
        
        
        # 3. Now create the terms, saving the UM call parameters for each ME.
        #    This means first identifying the kind of term (single atom, 2-body interaction, hopping),
        #    and then calling the appropriate function.
        
        # ***To start with, let's just create the single atom terms (on diagonal)
        # Definitions are in UM.Hterm, but the key question is:
        # if theUM.termsList[theTermNum].element1==None --> it's a single-atom term
        # otherwise check theUM.termsList[theTermNum].hop (True/False) to see if it's a hopping term
        
        molMats=[] # molMats[termNum][distortNum][sparseMatNum][if_mat_0_if_call_1]
        for theTerm in self.UM.termsList:
            molMats+=[self._makeMolMats(theTerm,molPl,theConfigsWrapper)]
            
        self.hmatTermLists+= [molMats] #gives molMats[molNum][termNum][distortNum][sparseMatNum][if_mat_0_if_call_1]
        
        #*** the returned list will look like matsList[distortionNum][distortionTypeNum][HmatListNum] = np.asarray[[sOrbInd, sOrbInd, E_s]]
        return []
        
    
    def _makeMolMats(self,theTerm,molPl,theConfigsWrapper):
        """Call the correct function to call to create the matrices for a term
        """
        
        termType=self._termType(theTerm)
        
        if termType==0: # if it's a single atom term
            return self._makeSingleAtomHmatTerms(theTerm,molPl,theConfigsWrapper)
                # This returns a single list of terms, which applies for all distortions
                
        elif termType==1:
            return self._make2AtomCoulombHmatTerms(theTerm,molPl,theConfigsWrapper)
                # This returns list[distortBondNum][distortionNum_0_thru_4][list_ind0_ind1_matrixElement2]
                
        elif termType==2:
            return None
        
    def _make2AtomCoulombHmatTerms(self,theTerm,molPl,theConfigsWrapper):
        """
        theConfigsWrapper.elementsLists[molPl]
        self.pairsList[molPl][distortBondNum][distortionNum_0_thru_4][pairNum][atom_1_or_2_or_dist]
        self.distortPairsList[moleculeNum][distortBondNum][distortionNum_1_thru_4][pairNum][atom_1_or_2_or_dist]
        theTerm.element0
        theTerm.max
        
        returns list[distortBondNum][distortionNum_0_thru_4][MEnum][list_ind0_ind1_matrixElement2]
        
        where [MEnum][list_ind0_ind1_matrixElement2] defines a sparse matrix
        """
        coulombTermsList = []
        
        for chosenBond in self.pairsList[molPl]:
            bonds = []
            for distortion in chosenBond:
                distortions = []
                for pair in distortion:
                    matElements = []
                    if pair[2] < theTerm.maxDist: # correct name?? # check if the pair distance is correct
                        if theConfigsWrapper.elementsList[molPl][pair[0]] == theTerm.element0 && theConfigsWrapper.elementsList[molPl][pair[1]] == theTerm.element1: # orbital symmetry check
                            matElement = theTerm.curve.readVal(pair[2])
                            pert0 = self.makeCFpert(pair[0], matElement, theUM.termsList[7].term[1])
                            pert1 = self.makeCFpert(pair[1], matElement, theUM.termsList[7].term[1]) # perturb the orbitals for both atoms in the pair
                            matElements += [pert0, pert1]
                     distortions += [matElements]
                bonds += [distortions]
         coulombTermsList += [bonds]
        
        return coulombTermsList
        
        #loop through pairsList -- for each pair:
        # (1) check if the distance is OK (<theTerm.max) 
        # (2) is the atom symmetry OK? is the first atom element0 and the 2nd atom element1
        # (3) identify the ME: theTerm.curve.readVal(theDistance)
        #symmetry???
        # (4) plug self.pairsList[molPl][distortBondNum][distortionNum_0_thru_4][pairNum], the ME, and the 
        # orbital symmetry theUM.termsList[7].term[1] into a new function "self.makeCFpert"
        # (5) reverse the order of the atoms and run #2-4 again
        
        #***check that _makeSingleAtomHmatTerms is compatible with distortList format
        
        return None
    
        
    def _makeSingleAtomHmatTerms(self,theTerm,molPl,theConfigsWrapper):
        """Referenced variables:
        theConfigsWrapper.elementsLists[molPl]
        self.hmatIndex
        self.UM.termsList[termNum].element0
        orbital symmetry: self.UM.termsList[theTermNum].term[0]
        """
        #first find the elements from theConfigsWrapper.elementsLists[molPl] that match self.UM.termsList[termNum].element0
        
        termsList = []
        for atomNum in range(len(theConfigsWrapper.elementsLists[molPl])):
            if theConfigsWrapper.elementsLists[molPl][atomNum]==theTerm.element0:
                #find the index
                orbSym= self.UM.getOrbSymNum(theTerm.element0,theTerm.term[0])
                
                # .orbSyms
                theIndex=self.hmatIndex[molPl][atomNum][orbSym]
                
                termsList += [[theIndex, theIndex,1],[]]
        
        termsList = [termsList]*5
        
        return termsList
        
        #then find the index for the selected orbital symmetry, and add a single list term [[indexPl,indexPl,1], []]
        # later, the '1' will be multiplied onto self.UM.termsList[theTermNum].val every time a hamiltonian is created
        
        #no
        
    def _termType(self, theTerm):
        """Returns an integer representing the type of Hamiltonian term referenced
        0: single atom
        1: Coulomb interaction term
        2: hopping term
        
        """
        if theTerm.element1 == None:
            return 0
        if theTerm.hop:
            return 2
        
        return 1
        
    
    def _makeHamiltonianIndex (self,theConfigsWrapper,theMoleculeNum):
        """ Define the initial Hamiltonian index for each [atom][orbital] within a single molecule (list of lists of integers)
        """

        molecule = theConfigsWrapper.elementsLists[theMoleculeNum]
        
        atomOrbTotals = []
        molOrbsIndexList=[]
        indexPl=0
        for atom in molecule:
            for element in self.UM.elementList:
                if atom == element:
                    atomInds = []
                    for orbital in self.UM.orbSyms[self.UM.elementList.index(element)]:
                        atomInds += [indexPl]
                        indexPl += orbital[1]
                    break # found the element in the UM
                    
            molOrbsIndexList += [atomInds]
            
        return molOrbsIndexList
        
       
    def findPairs(self, theConfigsWrapper, maxDist, molPl):
        #Loops through every atom pair in every distortion on every chosen bond in every molecule to find pairs under maxDist.

        #Returns bondPairs[distortBondNum][distortionNum_0_thru_4][pairNum][atom_1_or_2_or_dist]
        
        #***Partially tested.  Dimensions and output look reasonable.

        bondPairs = []
        for chosenBondIndex in range(2,len(theConfigsWrapper.distortLists[molPl])): #needs to be indented back
            singleBondPairs=[]
            for distortionIndex in range(len(theConfigsWrapper.distortLists[molPl][chosenBondIndex])):
                distortionPairs = []
                for indexOne in range(len(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex])):
                    for indexTwo in (range(indexOne+1, len(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex]))):
                             dist = np.linalg.norm(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexOne] - theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexTwo])
                             if dist <= maxDist:
                                 distortionPairs += [[indexOne, indexTwo, dist]]
                singleBondPairs += [distortionPairs]
            
            bondPairs += [singleBondPairs]
            
        return bondPairs
                             
        
# =============================================================================
#         self.pairs_list=[]
#         for ii in range(len(self.atoms_list)):
#             for jj in range(ii+1,len(self.atoms_list)):
#                 theDist=np.linalg.norm(self.coords_arry[jj,:]-self.coords_arry[ii,:])
#                 if theDist <= maxDist:
#                     self.pairs_list+=[[ii,jj,theDist]]    
# =============================================================================   
    
    def getElNums(self, theConfigsWrapper):
        """Uses the universal model and configs to generate a list of electron number for each configuration.
        This will reference self.UM.electronsPerAtom and self.UM.elementList
        
        Returns a list of electron number for each configuration: List of lists with length 5 for 4 distorted + 1 original configuration.
        """
        configSortLists = []
        for molecule in theConfigsWrapper.elementsLists: # loop through all molecules
            atomSortLists = []
            for element in self.UM.elementList:
                  atomSortLists += [[atom for atom in molecule if atom == element]] 
            configSortLists += [atomSortLists]
        
        electronNum = []
        for molecule in configSortLists:
            electronSum = 0
            for index in range(len(molecule)):
                electronSum += len(molecule[index]) * self.UM.electronsPerAtom[index]
            
            electronNum += [[electronSum]]
            
        electronsPerConfig = [[electronTotal for i in range(5)] for electronTotal in electronNum]    
        
        return electronsPerConfig
        #***
        
        
    def save(self,fileName):
        pass

    def makeHmats(self, theUM):
        pass
