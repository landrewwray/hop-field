#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022
@author: lawray

Develpment notes (Aug 7 2022):
    1. Hopping terms are not yet implemented in makeHmat
    2. Some Hamiltonian terms can likely be merged.  It would be good to implement this 
    in a future update.
    3. makeHmat is clearly too slow, and will need to be optimized.  It would be good to try numba on SigmoidCurve
    
Testing notes (Aug 7 2022):
    -All non-hopping terms have been tested for CH and C_2 dummy molecules separated along the x-axis

@author: lawray
"""

import numpy as np
from scipy import sparse
# import numba

# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
# import warnings
# warnings.resetwarnings()
# warnings.simplefilter('ignore')
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaWarning)

# import logging
# logger = logging.getLogger("numba")
# logger.setLevel(logging.ERROR)

resolutionLimit=1e-5  # Angstroms; minimum difference treated as equivalent, for the purposes of 
                      # condensing Hamiltonian terms lists

class SymTerms:
    def __init__(self, theUM, theConfigsWrapper):
        """
        theUM is a UniversalModel.  For this init, the Hamiltonian symmetries
        will matter, but the parameters multiplied onto them will not
        
        theConfigsWrapper is a list of atomic configurations, elements, etc
        """
        
        self.UM = theUM  # to remember they types of Hamiltonian terms
        self.electronNums=self.getElNums(theConfigsWrapper)  #***
        
        self._makeOrbDefs # define orbital vectors

        self.MElists=[]   # initialized by makeMElist
        self.pairsList=[] # initialized by makeMElist
        self.hmatIndex=[] # values created inside makeMElist
        self.hmatTermLists=[] # values created inside makeMElist
        
        # molPl=0
        # print("Molecule #" + str(molPl+1) + ": Hamiltonian pre-config")
        # self.MElists += self.makeMElist(theConfigsWrapper,molPl)
        
        for molPl in range(len(theConfigsWrapper.distortLists)): # ***fix the list name!
            print("Molecule #" + str(molPl+1) + ": Hamiltonian pre-config")
            self.MElists += self.makeMElist(theConfigsWrapper,molPl)
    
    def _makeOrbDefs(self):
        self.orbDefs = {'p': {'sigma': np.asarray([[0,0,1]]),'pi': np.asarray([[1,0,0],[0,1,0]])}, 's': {'sigma': np.asarray([[1]])}, 'sp': {'sigma': np.asarray([[1,0,0,1]])*2**-0.5}}
            
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
        self.pairsList += [self.findPairs(theConfigsWrapper, molPl)]
        
        
        # 3. Now create the terms, saving the UM call parameters for each ME.
        #    This means first identifying the kind of term (single atom, 2-body interaction, hopping),
        #    and then calling the appropriate function.
        
        # ***To start with, let's just create the single atom terms (on diagonal)
        # Definitions are in UM.Hterm, but the key question is:
        # if theUM.termsList[theTermNum].element1==None --> it's a single-atom term
        # otherwise check theUM.termsList[theTermNum].hop (True/False) to see if it's a hopping term
        
        molMats=[] # molMats[termNum][distortNum][sparseMatNum][if_mat_0_if_call_1]
        self._makeOrbDefs()
        termPl=0
        print("Term index: ", end = "")
        for theTerm in self.UM.termsList:
            print(str(termPl) + "/" + str(len(self.UM.termsList)-1), end = "")
            
            molMats+=[self._makeMolMats(theTerm,termPl,molPl,theConfigsWrapper)]
            termPl+=1
            if termPl<len(self.UM.termsList)-1:
                print(" .. ", end="")
            else:
                print()
            
        self.hmatTermLists+= [molMats] #gives self.hmatTermLists[molNum][termNum][distortedBondNum][distortionType_0thru4][sparseMatNum][if_mat_0_if_call_1]
        
        
        #*** the returned list will look like matsList[distortionNum][distortionTypeNum][HmatListNum] = np.asarray[[sOrbInd, sOrbInd, E_s]]
        return []
        
    
    def _makeMolMats(self,theTerm,termPl,molPl,theConfigsWrapper):
        """Call the correct function to call to create the matrices for a term
        """
        
        if theTerm.termType==0: # if it's a single atom term
            return self._makeSingleAtomHmatTerms(theTerm,molPl,theConfigsWrapper)
                # This returns a single list of terms, which applies for all distortions
                
        elif theTerm.termType==1: #Need to address 'E' energies for system-wide energy increase!
            # if theTerm.term[2] != 'E':
            return self._make2AtomCoulombHmatTerms(theTerm,termPl,molPl,theConfigsWrapper)
                # This returns list[distortBondNum][distortionNum_0_thru_4][list_ind0_ind1_matrixElement2]
                
        elif theTerm.termType==2:
            return self._makeHoppingTerms(theTerm,termPl, molPl, theConfigsWrapper) # check output!
        
    # @numba.jit
    def _makeHoppingTerms(self, theTerm, termPl, molPl, theConfigsWrapper):
        orbSym=theTerm.term[1:4] # for calling makeHop #NOTE: Not all theTerm.term arrays have ['s'/'p'/'sp', 'sigma'/'pi'] for the indices theTerm.term[1:3]
        hoppingTermsList = []
        for chosenBond in self.pairsList[molPl][termPl]:
            bonds = []
 
            for distortion in chosenBond:
                distortions = []
                for pair in distortion:
                    pair_v2=self.makeSwappedPair(pair)
                    complete_ME = False 
                    
                    # if pair[2] < self.UM.maxDist: # already checked in pairsList creation
                    if theConfigsWrapper.elementsLists[molPl][pair[0]] == theTerm.element0 and (theConfigsWrapper.elementsLists[molPl][pair[1]] == theTerm.element1): # orbital symmetry check
                        
                        hop0 = self.makeHop(pair, orbSym, molPl, theConfigsWrapper, theTerm)
                        distortions += [hop0]
                        if (theTerm.element0 == theTerm.element1) and orbSym[1] != orbSym[2]:
                            complete_ME = False # the reversed hopping scenario must be added in
                        else: 
                            complete_ME = True # done with this pair
                            
                    if complete_ME == False:
                        if theConfigsWrapper.elementsLists[molPl][pair_v2[0]] == theTerm.element0 and (theConfigsWrapper.elementsLists[molPl][pair_v2[1]] == theTerm.element1): # orbital symmetry check
                        
                            hop1 = self.makeHop(pair_v2, orbSym, molPl, theConfigsWrapper, theTerm)
                            distortions += [hop1]
                        
                bonds += [distortions]
            hoppingTermsList += [bonds]
        return hoppingTermsList
        # coulombTermsList[distortedBondNum][distortion_type][pair(not ordered)][0_sparseMat____1_distance_call_argument]
        
    # @numba.jit
    def makeHop(self,thePair,orbSym,molPl, theConfigsWrapper, theTerm):
        # only tested for high-symmetry geometries; gives the correct sign for all orbital
        # combinations and +/- orientation reversals.
        # print(orbSym)
        rotMat1 = self.singleRot(thePair[3],orbSym[1])
        if orbSym[2] != orbSym[1]:
            rotMat2 = self.singleRot(thePair[3],orbSym[2])
        else: 
            rotMat2 = rotMat1
        
        firstOrbVector=self.orbDefs[orbSym[1]][orbSym[0]] # Use two orbital definitions
        secondOrbVector=self.orbDefs[orbSym[2]][orbSym[0]]
        
        matDimRow = firstOrbVector.shape[1]    # width of a row
        matDimCol = secondOrbVector.shape[1]   # width of a col
        
        # Now build the term matrix
        theMat=np.zeros((matDimCol,matDimRow))
        for orbNum in range(firstOrbVector.shape[0]):
            theMat += rotMat2 @ (secondOrbVector[[orbNum],:].T @ firstOrbVector[[orbNum],:]) @ rotMat1.T # eg: |s><p|
            
        # *** Now convert to a sparse matrix and add the correct index for thePair[0] and indexOrb from self.hmatIndex
        indexOrbRow=orbSym[1][0] # turn 'sp' into 's' for indexing purposes
        
        orbitalRow = self.UM.getOrbSymNum(theConfigsWrapper.elementsLists[molPl][thePair[0]], indexOrbRow) # default for 's' orbital
        indexOrbCol=orbSym[2][0] # turn 'sp' into 's' for indexing purposes
        orbitalCol = self.UM.getOrbSymNum(theConfigsWrapper.elementsLists[molPl][thePair[1]], indexOrbCol) # default for 's' orbital
        theMatSparse = sparse.coo_matrix(theMat)
        
        # note that the above has used 'row' to indicate the column indices within a row,
        # which is reversed relative to the coo convention:

        theMatSparse.col += self.hmatIndex[molPl][thePair[0]][orbitalRow] # sparse matrix elements with corrected row and column indices based on atom and orbital
        theMatSparse.row += self.hmatIndex[molPl][thePair[1]][orbitalCol]
        theMatSparse.resize((self.hmatIndex[molPl][-1],self.hmatIndex[molPl][-1]))
        
        # Add reverse hopping:
        if theMatSparse.data.dtype=='complex':
            theMatSparse += theMatSparse.T.conj
            # sparse.vstack(theMatSparse, theMatSparse.T.conj)
        else:
            theMatSparse += theMatSparse.T
               
        # except:
        #     print(orbSym)
        #     print(thePair)
        #     print(self.hmatIndex[molPl][thePair[0]][orbitalRow])
        #     print(self.hmatIndex[molPl][thePair[1]][orbitalCol])
            
        #     print(self.hmatIndex[molPl])
        #     print(theMat)
            
        #     print(self.hmatIndex[molPl][-1])
        #     print(theMatSparse)
        #     print(theMatSparse.shape)
            
        #     import sys
        #     sys.exit()
        
        theMatSparse=sparse.coo_matrix(theMatSparse)

        
        # theMatSparse.row += self.hmatIndex[molPl][thePair[0]][orbitalRow] # sparse matrix elements with corrected row and column indices based on atom and orbital
        # theMatSparse.col += self.hmatIndex[molPl][thePair[1]][orbitalCol]    
        # this is the ME call:  matElement = theTerm.curve.readVal(thePair[2])
        return [theMatSparse, thePair[2]]  #return the sparse matrix and the distance needed for the call
       
    # @numba.jit
    def _make2AtomCoulombHmatTerms(self,theTerm,termPl,molPl,theConfigsWrapper):
        """***NOTE: theME needs to be looked up dynamically later, not multiplied onto these matrices!!!
        
        theConfigsWrapper.elementsLists[molPl]
        self.pairsList[molPl][distortBondNum][distortionNum_0_thru_4][pairNum][atom_1_or_2_or_dist]
        self.distortPairsList[moleculeNum][distortBondNum][distortionNum_1_thru_4][pairNum][atom_1_or_2_or_dist]
        theTerm.element0
        theTerm.max
        
        returns list[distortBondNum][distortionNum_0_thru_4][MEnum][list_ind0_ind1_matrixElement2]
        
        where [MEnum][list_ind0_ind1_matrixElement2] defines a sparse matrix
        """
        
        #loop through pairsList -- for each pair:
        # (1) check if the distance is OK (<theTerm.max) 
        # (2) is the atom symmetry OK? is the first atom element0 and the 2nd atom element1
        # (3) identify the ME: theTerm.curve.readVal(theDistance)
        #symmetry???
        # (4) plug self.pairsList[molPl][distortBondNum][distortionNum_0_thru_4][pairNum], the ME, and the 
        # orbital symmetry theUM.termsList[7].term[1] into a new function "self.makeCFpert"
        # (5) reverse the order of the atoms and run #2-4 again
        
        orbSym=theTerm.term[1:3] # for calling makeCFpert #NOTE: Not all theTerm.term arrays have ['s'/'p'/'sp', 'sigma'/'pi'] for the indices theTerm.term[1:3]
        coulombTermsList = []
        for chosenBond in self.pairsList[molPl][termPl]:
            bonds = []
            distortionPl=-1
            for distortion in chosenBond:
                distortionPl+=1
                distortions = []
                for pair in distortion:
                    pair_v2=self.makeSwappedPair(pair) # Swap elements 0 and 1, and reverse the direction of the vector

                    # if pair[2] < self.UM.maxDist:  # already checked in pairsList creation
                    if theConfigsWrapper.elementsLists[molPl][pair[0]] == theTerm.element0 and (theConfigsWrapper.elementsLists[molPl][pair[1]] == theTerm.element1): # orbital symmetry check
                        
                        pert0 = self.makeCFpert(pair, orbSym, molPl, theConfigsWrapper, theTerm, distortionPl)
                        # pert1 = self.makeCFpert(pair, orbSym, molPl) # perturb the orbitals for both atoms in the pair
                        distortions += [pert0]
                    
                    if theConfigsWrapper.elementsLists[molPl][pair_v2[0]] == theTerm.element0 and (theConfigsWrapper.elementsLists[molPl][pair_v2[1]] == theTerm.element1): # orbital symmetry check
                        
                        skipFlip = (theTerm.element0 == theTerm.element1) and (orbSym[1] == 'E')
                        if not skipFlip:  #avoid double-couning E-terms between the same element
                            pert1 = self.makeCFpert(pair_v2, orbSym, molPl, theConfigsWrapper, theTerm, distortionPl)
                            # pert1 = self.makeCFpert(pair, orbSym, molPl) # perturb the orbitals for both atoms in the pair
                            distortions += [pert1]
                        
                bonds += [distortions]        
            coulombTermsList += [bonds]
            
        return coulombTermsList  
        # coulombTermsList[distortedBondNum][distortion_type][pair(not ordered)][0_sparseMat____1_distance_call_argument]

    # @numba.jit
    def makeSwappedPair(self,thePair):
        """Swap the order of atoms in the pair. 
        
        Swap elements 0 and 1, and reverse the direction of the vector
        """
        
        return([thePair[1], thePair[0], thePair[2], -1*thePair[3]])

    # @numba.jit
    def makeCFpert(self,thePair,orbSym,molPl, theConfigsWrapper, theTerm, distortionPl):
        """***Not yet tested! Create the orbital perturbation matrix elements
        ***NOTE: theME needs to be looked up dynamically later, not multiplied onto these matrices!!!
        
        Possible orbSym values are:
            ['s'/'p'/'sp', 'sigma'/'pi'] : see init matrices
            'E': on-diagonal perturbation for all electrons, not just on the paired atoms
                (i.e. this is a classical energy term for the atomic configuration)
            
        Returns a list representing a Hermetian sub-matrix
        """
        
        if orbSym[1] != 'E':
            rotMat = self.singleRot(thePair[3],orbSym[1])
    
            theOrbVectors=self.orbDefs[orbSym[1]][orbSym[0]]
            matDim = theOrbVectors.shape[1]
            
            # Now build the term matrix
            theMat=np.zeros((matDim,matDim))
            for orbNum in range(theOrbVectors.shape[0]):
                theMat += rotMat @ (theOrbVectors[[orbNum],:].T @ theOrbVectors[[orbNum],:]) @ rotMat.T   
            
            # *** Now convert to a sparse matrix and add the correct index for thePair[0] and indexOrb from self.hmatIndex
            indexOrb=orbSym[1][0] # turn 'sp' into 's' for indexing purposes
            orbital = self.UM.getOrbSymNum(theConfigsWrapper.elementsLists[molPl][thePair[0]], indexOrb) # default for 's' orbital
    
            theMatSparse = sparse.coo_matrix(theMat)
            
            theMatSparse.row += self.hmatIndex[molPl][thePair[0]][orbital] # sparse matrix elements with corrected row and column indices based on atom and orbital
            theMatSparse.col += self.hmatIndex[molPl][thePair[0]][orbital]
            theMatSparse.resize((self.hmatIndex[molPl][-1],self.hmatIndex[molPl][-1]))
    
            # this is the ME call:  matElement = theTerm.curve.readVal(thePair[2])
            return [theMatSparse, thePair[2]]  #return the sparse matrix and the distance needed for the call
       
        else:  # if it's an 'E' term
            theME=sparse.coo_matrix(np.eye(self.hmatIndex[molPl][-1]))
            theME.data /= self.electronNums[molPl][distortionPl]
            return [theME, thePair[2]]
       
         # one more index needed: [***last index is 0 for indexOrb=='s' and 1 for indexOrb=='p']

    # @numba.jit
    def singleRot(self,axisDir,theOrb):        
        #returns a rotation matrix that acts from the left
        # ***not yet tested for sp orbital

        if theOrb == 's':
            return np.asarray([1])
        if theOrb == 'p' or theOrb == 'sp':
            randomOrientation=np.random.rand(3)
            zPrime = axisDir/np.linalg.norm(axisDir)
            xPrime = np.cross(randomOrientation,zPrime)
            yPrime = np.cross(xPrime,zPrime)
            
            xPrime/=np.linalg.norm(xPrime)
            yPrime/=np.linalg.norm(yPrime)
            
            theMat=  zPrime.reshape(3,1) @ np.asarray([[0, 0, 1]]) + xPrime.reshape(3,1) @ np.asarray([[1, 0, 0]]) + yPrime.reshape(3,1) @ np.asarray([[0, 1, 0]])            
        
            if theOrb == 'p': # == [0, 0, 1]
                return theMat
            else: #theOrb == 'sp' == [1, 0, 0, 1]*2**-1
                # R @ |sp> @ <sp| @R.T
                finalMat=np.zeros((4,4))
                finalMat[0,0]=1
                finalMat[1:,1:]=theMat
                return finalMat
            
        
        
    
    # def orbitalTransform(self,theDir,orbSym):
        
        
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
                
                numOrbs=len(self.orbDefs[theTerm.term[0]]['sigma'][0]) # read the orbital subbasis size
                
                for orbAdd in range(numOrbs):
                    termsList += [[theIndex+orbAdd, theIndex+orbAdd,1]]
        
        # if termsList ==[]:
        #     termsList = [[[]]]
        
        return termsList
        
        #then find the index for the selected orbital symmetry, and add a single list term [[indexPl,indexPl,1], []]
        # later, the '1' will be multiplied onto self.UM.termsList[theTermNum].val every time a hamiltonian is created
        
        #no
    
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
            
            # print('here')
            # print(atom)
            molOrbsIndexList += [atomInds]
        
        molOrbsIndexList += [indexPl]   # molOrbsIndexList[-1] is the dimensionality of the Hamiltonia
        
        return molOrbsIndexList
        
       
    def findPairs(self, theConfigsWrapper, molPl):
        """
        Loops through every atom pair in every distortion on every chosen bond 
        in the molPl molecule to the sparation vector for pairs separated by <maxDist. 
        For convenience when condensing terms, only pairs separated by <maxDist 
        in the undistorted molecular configuration are included.

        Returns bondPairs[termNum][distortBondNum][distortionNum_0_thru_4][pairNum][atom0_or_atom1_or_dist]
        
        The axisDir vector points from atom0 (the 'acted on' atom) to atom1.
        
        ***Partially tested.  Dimensions and output look reasonable.
        
        """
        
        # first find the allowed pairs
        numAtoms = len(theConfigsWrapper.distortLists[molPl][2][0])
        
        goodPairs=[[] for term in self.UM.termsList]  #identify valid pairs in the undistorted structure
        for indexOne in range(numAtoms):
            for indexTwo in range(indexOne+1, numAtoms):
                dist = np.linalg.norm(theConfigsWrapper.distortLists[molPl][2][0][indexOne] - theConfigsWrapper.distortLists[molPl][2][0][indexTwo])
                
                for termPl in range(len(self.UM.termsList)):
                    
                    if self.UM.termsList[termPl].curve != None:
                        maxDist = self.UM.termsList[termPl].curve.dMax
                        
                        if dist<maxDist:
                            goodPairs[termPl] += [[indexOne,indexTwo]]
        
        bondPairs = [[] for term in self.UM.termsList]
        for termPl in range(len(self.UM.termsList)):
            for chosenBondIndex in range(2,len(theConfigsWrapper.distortLists[molPl])): #needs to be indented back
                singleBondPairs=[]
                for distortionIndex in range(len(theConfigsWrapper.distortLists[molPl][chosenBondIndex])):
                    distortionPairs = []
                    for pair in goodPairs[termPl]:
                        axisDir = theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][pair[1]] - theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][pair[0]]
                        dist = np.linalg.norm(axisDir)
                            
                        distortionPairs += [[pair[0], pair[1], dist, axisDir]]
                    singleBondPairs += [distortionPairs]
                
                bondPairs[termPl] += [singleBondPairs]
            
        return bondPairs
        
        # bondPairs = []
        # for chosenBondIndex in range(2,len(theConfigsWrapper.distortLists[molPl])): #needs to be indented back
        #     singleBondPairs=[]
        #     for distortionIndex in range(len(theConfigsWrapper.distortLists[molPl][chosenBondIndex])):
        #         distortionPairs = []
        #         for indexOne in range(len(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex])):
        #             for indexTwo in (range(indexOne+1, len(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex]))):
        #                       dist = np.linalg.norm(theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexOne] - theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexTwo])
        #                       axisDir = theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexTwo] - theConfigsWrapper.distortLists[molPl][chosenBondIndex][distortionIndex][indexOne]
        #                       if dist <= maxDist:
        #                           distortionPairs += [[indexOne, indexTwo, dist, axisDir]]
        #         singleBondPairs += [distortionPairs]
            
        #     bondPairs += [singleBondPairs]
            
        # return bondPairs
                             
        
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
        

    def makeHmat(self, theUM, molPl, whichMat):
        """Creates a single Hamiltonian matrix. Note that there are ~13*(4*5+1)=273 unique Hamiltonians
        whichMat specifies the matrix as: (distortedBondNum, distortionType_0thru4)
        
        returns: theHmat  (Numpy array)
        
        NOTE: This function exists for debugging purposes only, and is currently only implemented for 
        orbital energies and Coulomb terms (not hopping). For rapid full Hamiltonian creation, see the 
        monte_carlo package
        
        Optimization:
        If sparse implementation is not enough, use Numba:
            1. Will probably want to unpack so that sparse matrix elements and calls are just one long list
            2. Loop by term type and call sub-functions that only see the relevant UM term data
            
        Debug: -- sp terms are not showing up?
            
        """
        # use:
        # self.hmatTermLists[molPl][termNum][distortedBondNum][distortionType_0thru4][sparseMatNum][if_mat_0_if_call_1]
        # self.UM.termsList[number].termType
        # self.hmatIndex[molPl][-1]   --- the matrix dimensionality
        
        theHmat=np.zeros((self.hmatIndex[molPl][-1],self.hmatIndex[molPl][-1]))
        
        termNum=0
        for theTerm in theUM.termsList:
            if theTerm.termType == 0: # orbital energy
                for listPl in range(len(self.hmatTermLists[molPl][termNum])):
                    theIndex=self.hmatTermLists[molPl][termNum][listPl][0]
                    theHmat[theIndex,theIndex] += theTerm.val
                    
            elif theTerm.termType == 1: # Coulomb
                theHmat += self.makeHmatCoulombTerm(theUM, molPl, whichMat, termNum)
            else:  # hopping term
                pass
            
            termNum += 1
            
        return theHmat
    
    def makeHmatCoulombTerm(self, theUM, molPl, whichMat, termNum):

        # self.hmatTermLists[molPl][termNum][whichMat[0]][whichMat[1]][sparseMatNum][if_mat_0_if_call_1]   
        
        theHmat=0
        numSubMats=len(self.hmatTermLists[molPl][termNum][whichMat[0]][whichMat[1]])
        
        for addMat in self.hmatTermLists[molPl][termNum][whichMat[0]][whichMat[1]]:
            theHmat += addMat[0] * theUM.termsList[termNum].curve.readVal(addMat[1])
        
        return theHmat

def mergeLikeMats(symTerms):
    """Merge all symmetry matrices with the same distance call for each
    mol/term/bond/distortionType permutation.
    
    ***Output looks correct --- need to check with full Hmat creation
    
    ***Error: This occasionally results in different numbers of terms for 
    different distPl values, as seen in:
    # print(len(symMats.hmatTermLists[1][3][2][3]))
    # print(len(symMats.hmatTermLists[1][3][2][2]))
    
    This violates an assumption of make_monte_carlo_lists
    """
    
    for molPl in range(len(symTerms.hmatTermLists)):
        for termNum in range(len(symTerms.hmatTermLists[molPl])):
            for distortedBondNum in range(len(symTerms.hmatTermLists[molPl][termNum])):
                for distortionType in range(len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum])):
                    if type(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType]) == list:
                        #if it's a curve-based term, then try to merge:
                        for termPl1 in range(len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType])):
                            termPl2=termPl1+1

                            while termPl2 < len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType]):
                                if abs(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl1][1] - symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl2][1]) < resolutionLimit:
                                    symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl1][0] += symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl2][0]
                                    
                                    # make sure it's still a COO matrix (format will convert
                                    # to CSR upon matrix addition in scipy 1.7.3):
                                    symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl1][0] = sparse.coo_matrix(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType][termPl1][0])
                                                                                                                                             
                                    symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionType].pop(termPl2)
                                else:
                                    termPl2+=1
                                    
    #self.hmatTermLists[molNum][termNum][distortedBondNum][distortionType_0thru4][sparseMatNum][if_mat_0_if_call_1]

class MCwrapper:
    pass

def make_monte_carlo_lists(symTerms):
    """Arrange the Hmat information in term/mol/bond/distortionType[numpyArray] form.
    The last distortionType index is reserved for matrix elements that are common to all
    distortions, whereas individual [distortionType] indices only list terms unique to a 
    particular distortion.
    
    The purpose of these unpacked data structures is to enable straighforward optimization
    of Hamiltonian creation in numba/cython
    
    ***TEST!!!!***
    """
    intdt=np.int32
    floatdt=np.float64  #consider float32 if memory is tight...
    
    callDist=[]  # 1D (later convert to numpy)
    firstInd=[]  # 1D (later convert to numpy)
    lastInd=[]   # 1D (later convert to numpy)
    callFirstLast=[]
    symmetryRowArray=[]  # (int32) --> numpy
    symmetryColArray=[]
    symmetryIndsArray =[] # this will store the row and col inds after numpy conversion
    symmetryValArray=[]  # (float) --> numpy
    
    isSingleAtom = np.zeros(len(symTerms.hmatTermLists[0]))
    
    for termNum in range(len(symTerms.hmatTermLists[0])): # all molecules have the same number of terms
        callDist += [[]]  # [termNum]
        firstInd += [[]]  
        lastInd  += [[]]  
        callFirstLast += [[]]
        symmetryRowArray += [[]]  
        symmetryColArray += [[]]
        symmetryIndsArray += [[]]
        symmetryValArray += [[]] 
    
        for molPl in range(len(symTerms.hmatTermLists)):
            callDist[termNum] += [[]]  # [molPl]
            firstInd[termNum] += [[]]  
            lastInd[termNum]  += [[]]  
            callFirstLast[termNum]  += [[]]
            symmetryRowArray[termNum] += [[]] 
            symmetryColArray[termNum] += [[]]
            symmetryIndsArray[termNum] += [[]]
            symmetryValArray[termNum] += [[]] 
            
            # check here if a term is an orbital energy (not bond/distortion-dependent):

            # if type(symTerms.hmatTermLists[molPl][termNum][0][0]) == list: # if it's not an orbital energy
            if symTerms.UM.termsList[termNum].termType != 0: # if it's not an orbital energy
                for distortedBondNum in range(len(symTerms.hmatTermLists[molPl][termNum])):
                    callDist[termNum][molPl] += [[]]  # [distortedBondNum]
                    firstInd[termNum][molPl] += [[]]  
                    lastInd[termNum][molPl]  += [[]]  
                    callFirstLast[termNum][molPl]  += [[]]  
                    symmetryRowArray[termNum][molPl] += [[]] 
                    symmetryColArray[termNum][molPl] += [[]]
                    symmetryIndsArray[termNum][molPl] += [[]]
                    symmetryValArray[termNum][molPl] += [[]] 
                    
                    for distortionNum in range(1+len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum])):
                        #stand-alone loop to init distortionNum index, with an extra spot for distortion-independent terms
                        callDist[termNum][molPl][distortedBondNum] += [[]]  # [distortionNum]
                        firstInd[termNum][molPl][distortedBondNum] += [[]]  
                        lastInd[termNum][molPl][distortedBondNum]  += [[]] 
                        callFirstLast[termNum][molPl][distortedBondNum]  += [[]] 
                        symmetryRowArray[termNum][molPl][distortedBondNum] += [[]] 
                        symmetryColArray[termNum][molPl][distortedBondNum] += [[]]
                        symmetryIndsArray[termNum][molPl][distortedBondNum] += [[]]
                        symmetryValArray[termNum][molPl][distortedBondNum] += [[]]
                    
                    indexPl_indep = 0
                    indexPl_dep = 0
                    for listPl in range(len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum][0])):
                        # this loop is over the final mat list (skipping distortionNum)
                        listItem_indep=symTerms.hmatTermLists[molPl][termNum][distortedBondNum][0][listPl]
                        
                        # check if each term is unchanged versus distortionType (reference resolutionLimit)

                        baseCall = symTerms.hmatTermLists[molPl][termNum][distortedBondNum][0][listPl][1]
                        isIndep=True
                        checkPl=1
                        while isIndep and checkPl<len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum]):
                            
                            if abs(baseCall-symTerms.hmatTermLists[molPl][termNum][distortedBondNum][checkPl][listPl][1]) > resolutionLimit:
                                isIndep = False #flag as a distortion-dependant term
                            checkPl += 1
                            
                        if isIndep:
                            # put these terms in the last distortion spot
                            callDist[termNum][molPl][distortedBondNum][-1] += [listItem_indep[1]]  
                            firstInd[termNum][molPl][distortedBondNum][-1] += [indexPl_indep]  
                            indexPl_indep+=len(listItem_indep[0].row)
                            lastInd[termNum][molPl][distortedBondNum][-1]  += [indexPl_indep]  
                            symmetryRowArray[termNum][molPl][distortedBondNum][-1] += listItem_indep[0].row.tolist()
                            symmetryColArray[termNum][molPl][distortedBondNum][-1] += listItem_indep[0].col.tolist()
                            symmetryValArray[termNum][molPl][distortedBondNum][-1] += listItem_indep[0].data.tolist()
                        else:
                            # insert distortion-dependent terms
                            fromIndexPl_dep = indexPl_dep
                            indexPl_dep += len(listItem_indep[0].row)
                            for distortionNum in range(len(symTerms.hmatTermLists[molPl][termNum][distortedBondNum])):
                                listItem_dep=symTerms.hmatTermLists[molPl][termNum][distortedBondNum][distortionNum][listPl]
                                
                                callDist[termNum][molPl][distortedBondNum][distortionNum] += [listItem_dep[1]]  
                                firstInd[termNum][molPl][distortedBondNum][distortionNum] += [fromIndexPl_dep]  
                                lastInd[termNum][molPl][distortedBondNum][distortionNum]  += [indexPl_dep]  
                                
                                symmetryRowArray[termNum][molPl][distortedBondNum][distortionNum] += listItem_dep[0].row.tolist()
                                symmetryColArray[termNum][molPl][distortedBondNum][distortionNum] += listItem_dep[0].col.tolist()
                                symmetryValArray[termNum][molPl][distortedBondNum][distortionNum] += listItem_dep[0].data.tolist()
                               
                    for distortionNum in range(len(symmetryValArray[termNum][molPl][distortedBondNum])):
                        #convert to numpy for list scenario
                        callDist[termNum][molPl][distortedBondNum][distortionNum] = np.asarray(callDist[termNum][molPl][distortedBondNum][distortionNum],dtype=floatdt)
                        callFirstLast[termNum][molPl][distortedBondNum][distortionNum] = np.asarray([firstInd[termNum][molPl][distortedBondNum][distortionNum],lastInd[termNum][molPl][distortedBondNum][distortionNum]], dtype=intdt) 
                        symmetryIndsArray[termNum][molPl][distortedBondNum][distortionNum] = np.asarray([symmetryRowArray[termNum][molPl][distortedBondNum][distortionNum],symmetryColArray[termNum][molPl][distortedBondNum][distortionNum]], dtype=intdt)
                        symmetryValArray[termNum][molPl][distortedBondNum][distortionNum] = np.asarray(symmetryValArray[termNum][molPl][distortedBondNum][distortionNum],dtype=floatdt)

            else:
                # now deal with the orbital energy terms
                isSingleAtom[termNum] = 1
                for listItems in symTerms.hmatTermLists[molPl][termNum]:
                    # callDist[termNum][molPl] += [[]] # not needed! All single atom list items have the same call
                    # firstInd[termNum][molPl] += [[]]  
                    # lastInd[termNum][molPl]  += [[]]  
                    symmetryRowArray[termNum][molPl] += [listItems[0]] 
                    symmetryColArray[termNum][molPl] += [listItems[1]]
                    symmetryValArray[termNum][molPl] += [listItems[2]] 
                
                # to numpy for the same-atom scenario
                symmetryIndsArray[termNum][molPl]= np.asarray([symmetryRowArray[termNum][molPl],symmetryColArray[termNum][molPl]],dtype=intdt)
                symmetryValArray[termNum][molPl] = np.asarray(symmetryValArray[termNum][molPl],dtype=floatdt)
                
                
    #record the size of the Hamiltonian matrices:
    
    
    # transfer to the wrapper
       # symmetryRowArray[termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
       # isSingleAtom[termNum]
    mcLists = MCwrapper()
    
    mcLists.callDist = callDist
    mcLists.callFirstLast = callFirstLast # [first0_last1,callNumber]
   
    mcLists.symmetryValArray = symmetryValArray
    
    mcLists.symmetryIndsArray = symmetryIndsArray # [row0_col1,indNumber]
    
    mcLists.isSingleAtom = isSingleAtom
    
    
    
    mcLists.hmatIndex = symTerms.hmatIndex # hmatIndex[molPl][-1] gives the Hmat size
    
    clumpTerms(mcLists) # Merge everything beneath [termPl] into large numpy arrays
    
    return mcLists

def clumpTerms(mcLists):
    """Everything for a given term can be clumped into a long numpy array! This will
    necessitate creating the following:
        
    callFirstLast: stores the last index for each distance call; special entries are
        -1: toggle to the next distPl
        -2: toggles to next bondPl and 1st distPl
        -3: next molPl and 1st bondPl+distPl
    matEnds: store the valList index range for each Hmat matrix. In each case, one can store multiple
    lists for a single matrix (e.g. the matrix-specific index range and the shared index range). For example:
        matEnds[0,:] = [0,100,  -1,  20, 100,  -2, ...]
        matEnds[1,:] = [20,150,  -1, 39, 150,  -2, ...]
        
        Special entries of matEnds are the same as for callFirstLast. The inclusion of
        these special entries in callFirstLast is primarily for debugging purposes, whereas 
        they are useful to guide Hmat stitching in matEnds.
        
        
    Note: make sure to crop out calls with no entries! (symmetry ME of 0 should 
    already be excluded -- check***)
    
    
    mcLists members after this update:  
        matEnds --> see above
        callFirstLast --> callFirstLast[termPl](callPl) -- updated to the format specified above 
                                                    (one 2D numpy array per term)
        callDist --> callDist[termPl](callPl) (one 1D numpy array per term)
        symmetryValArray --> symmetryValArray[termPl](ind) -- one 1D numpy array per term
        symmetryIndsArray --> symmetryIndsArray[termPl](ind) -- one 2D numpy array per term
        
        isSingleAtom --> unchanged
        hmatIndex --> unchanged
        
        
        
    To do:
    X-check if symmetryValArray[termPl][molPl] is a simple list for single atom terms
    X-define all terms
    -make sure the last distortPl is appropriately entered
    -check: what happens if there are no valid MEs for a given Hamiltonian term
    -remove the break points from callFirstLast_new after debugging -- and remove 
       the associated conditional from make_hmat.fillValList
    
    """
    
    
    callFirstLast_new = []
    callDist_new = []
    matEnds = []
    
    symmetryValArray_new = []
    symmetryIndsArray_new = []
    for termPl in range(len(mcLists.symmetryValArray)):
        # Init [callPl] terms
        currentValPl = 0

        matEnds += [[[],[]]] 
        callDist_new += [[]]
        callFirstLast_new += [[[],[]]] 
        
        # Init [ind] terms
        symmetryValArray_new += [[]]
        symmetryIndsArray_new += [[[],[]]] 
        
        for molPl in range(len(mcLists.symmetryValArray[termPl])):
            
            if not mcLists.isSingleAtom[termPl]:
                for bondPl in range(len(mcLists.symmetryValArray[termPl][molPl])):
                    numDistortions=len(mcLists.symmetryValArray[termPl][molPl][bondPl])
                    matEnds_tmp = [currentValPl]
                    
                    for distortPl in range(numDistortions):
                        #Now loop through the calls
                        symmetryValArray_new[termPl] += list(mcLists.symmetryValArray[termPl][molPl][bondPl][distortPl])
                        callDist_new[termPl] += list(mcLists.callDist[termPl][molPl][bondPl][distortPl])
                        symmetryIndsArray_new[termPl][0] += list(mcLists.symmetryIndsArray[termPl][molPl][bondPl][distortPl][0])
                        symmetryIndsArray_new[termPl][1] += list(mcLists.symmetryIndsArray[termPl][molPl][bondPl][distortPl][1])
                           #still missing matEnds and callFirstLast
                        
                        totNewInds=len(mcLists.symmetryValArray[termPl][molPl][bondPl][distortPl])
                        
                        # now filter out empty calls:
                        for callNum in range(len(mcLists.callFirstLast[termPl][molPl][bondPl][distortPl][0])):
                            # If there's something to add:
                            
                            numCallInds = mcLists.callFirstLast[termPl][molPl][bondPl][distortPl][1,callNum]-mcLists.callFirstLast[termPl][molPl][bondPl][distortPl][0,callNum]   
                            
                            if numCallInds > 0:    
                                callFirstLast_new[termPl][0] += [mcLists.callFirstLast[termPl][molPl][bondPl][distortPl][0,callNum] + currentValPl]
                                callFirstLast_new[termPl][1] += [mcLists.callFirstLast[termPl][molPl][bondPl][distortPl][1,callNum] + currentValPl]
 
                        currentValPl += totNewInds # track the number of matrix elements
                            
                        # All MEs have been entered.  Now insert break points
                        callFirstLast_new[termPl][0]+=[-1] #end of one distortion
                        callFirstLast_new[termPl][1]+=[-1] #end of one distortion
                        if distortPl<numDistortions-1:
                            matEnds_tmp += [currentValPl]    #track the end point of terms for each Hmat matrix
                        
                    # after the last distortion:
                    # Update matEnds:
                    commonMatEnds=[matEnds_tmp[-1],currentValPl]
                    for pl in range(len(matEnds_tmp)-1):
                        matEnds[termPl][0]+=[matEnds_tmp[pl]]
                        matEnds[termPl][1]+=[matEnds_tmp[pl+1]]
                        
                        matEnds[termPl][0]+=[commonMatEnds[0]]
                        matEnds[termPl][1]+=[commonMatEnds[1]]
                        
                        matEnds[termPl][0]+=[-1]
                        matEnds[termPl][1]+=[-1]
                    
                    #now correct the final break point (will be -2 or -3)
                    if bondPl == len(mcLists.symmetryValArray[termPl][molPl]) - 1:
                        
                        assert matEnds[termPl][0][-1] == -1, print('hmat_stitcher: Error in matEnds creation. Location: ' + str([termPl,molPl,bondPl,distortPl]))
                            
                        matEnds[termPl][0][-1] = -3
                        matEnds[termPl][1][-1] = -3
                        
                        assert callFirstLast_new[termPl][0][-1] == -1, print('hmat_stitcher: Error in callFirstLast conversion. Location: ' + str([termPl,molPl,bondPl,distortPl]))
                        
                        callFirstLast_new[termPl][0][-1] = -3 
                        callFirstLast_new[termPl][1][-1] = -3
                    else:
                        assert matEnds[termPl][0][-1] == -1, print('hmat_stitcher: Error in matEnds creation. Location: ' + str([termPl,molPl,bondPl,distortPl]))
                        assert callFirstLast_new[termPl][0][-1] == -1, print('hmat_stitcher: Error in callFirstLast conversion. Location: ' + str([termPl,molPl,bondPl,distortPl]))
                        
                        matEnds[termPl][0][-1] = -2
                        matEnds[termPl][1][-1] = -2
                        callFirstLast_new[termPl][0][-1] = -2 
                        callFirstLast_new[termPl][1][-1] = -2 
                        
            else: ######## single atom terms ########
            
                # If there's something to add:
                numInds = len(mcLists.symmetryValArray[termPl][molPl])
                
                symmetryValArray_new[termPl] += list(mcLists.symmetryValArray[termPl][molPl])
                symmetryIndsArray_new[termPl][0] += list(mcLists.symmetryIndsArray[termPl][molPl][0])
                symmetryIndsArray_new[termPl][1] += list(mcLists.symmetryIndsArray[termPl][molPl][1])
                callFirstLast_new[termPl][0] += [currentValPl]
                callFirstLast_new[termPl][1] += [currentValPl+numInds]
                # callDist/callDist_new is not needed for single atom terms

                matEnds[termPl][0] += [currentValPl]
                matEnds[termPl][1] += [currentValPl+numInds]
                
                # break points
                callFirstLast_new[termPl][0] += [-3]
                callFirstLast_new[termPl][1] += [-3]
                matEnds[termPl][0] += [-3]
                matEnds[termPl][1] += [-3]
                
                currentValPl+=numInds
                
        #now convert to numpy:
            
        symmetryValArray_new[termPl] = np.asarray(symmetryValArray_new[termPl], dtype=np.float64)
        symmetryIndsArray_new[termPl] = np.asarray(symmetryIndsArray_new[termPl], dtype=np.int32)
        callDist_new[termPl] = np.asarray(callDist_new[termPl], dtype=np.float64)
                
        callFirstLast_new[termPl] = np.asarray(callFirstLast_new[termPl], dtype=np.int32)
        matEnds[termPl] = np.asarray(matEnds[termPl], dtype=np.int32)
        
    #now overwrite the modified lists:
    # mcLists.symmetryValArray2 = symmetryValArray_new
    # mcLists.symmetryIndsArray2 = symmetryIndsArray_new
    # mcLists.callDist2 = callDist_new
    # mcLists.callFirstLast2 = callFirstLast_new

    mcLists.symmetryValArray = symmetryValArray_new
    mcLists.symmetryIndsArray = symmetryIndsArray_new
    mcLists.callDist = callDist_new
    mcLists.callFirstLast = callFirstLast_new
    mcLists.matEnds = matEnds
    
    return None
    
    
        
        