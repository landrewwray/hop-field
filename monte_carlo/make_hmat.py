#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:22:57 2022

@author: L. Andrew Wray

Uses numba acceleration to rapidly create and update hamiltonian matrices. Hamiltonian
matrix elements are defined from the current model parameters and from the 
model_creation symmetry lists.

TO DO (10/04/2022):
1. Separate out monte carlo fucntions
2. Evaluate the Hamiltonian term-by-term for a minimally complex molecule
"""

import copy
import numpy as np
from numba import njit, int32, float64
#from numba.experimental import jitclass

maxDisplacement = 50 # Units of the displacement size


@njit
def fillValList(symmetryValArray,callDistList,callFirstLast,termCurve):
    
    valList=np.zeros(symmetryValArray.shape[0])
    
    callNum=-1
    for firstLastPl in range(callFirstLast.shape[1]):
        if callFirstLast[1,firstLastPl] > 0: # if this is not a break point
            callNum += 1
            callVal=termCurve.readVal(callDistList[callNum])
            
            # print([callFirstLast[0,firstLastPl], callFirstLast[1,firstLastPl], len(valList)])
            for valPl in range(callFirstLast[0,firstLastPl],callFirstLast[1,firstLastPl]):  
                valList[valPl] = symmetryValArray[valPl]*callVal

    return valList

def generate_empty_HmatList(mcLists):
    """nested list of [molPl][bondPl][distPl]
    
    This implementation may have an extra empty slot in distPl -- CHECK! ***
    
    ***Test!!!***
    """
    
    #choose a non-single-atom term to identify the full dimensionality:
    nonSAterm=np.where(mcLists.isSingleAtom==0)[0][0]
    
    molPl=0
    bondPl=0
    hmatList=[[[np.zeros((mcLists.hmatIndex[molPl][-1],mcLists.hmatIndex[molPl][-1]))]]]
    
    for matNum in mcLists.matEnds[nonSAterm][0][:-1]: # ignore the terminal -3
        if matNum == -1:  # if it's a distortion breakPoint
            hmatList[molPl][bondPl] += [np.zeros((mcLists.hmatIndex[molPl][-1],mcLists.hmatIndex[molPl][-1]))]
        elif matNum == -2:  # if it's a bondPl breakPoint
            bondPl += 1
            hmatList[molPl] += [[np.zeros((mcLists.hmatIndex[molPl][-1],mcLists.hmatIndex[molPl][-1]))]]
        elif matNum == -3: # if it's a molPl breakPoint
            molPl += 1
            bondPl = 0
            hmatList += [[[np.zeros((mcLists.hmatIndex[molPl][-1],mcLists.hmatIndex[molPl][-1]))]]]

    return hmatList 


def makeHmatValList(termPl, mcLists, termCurves):
    """
    Create the new valList[termPl] of values to be added into the Hamiltonian. The current state will save 
    both the new valList and the last iteration, so that the valList[termPl]-oldValList[termPl] difference
    can be added into the Hamiltonian without needing to recreate the entire matrix.
    
    Note that the returned list is the same object as 'oldTermHmat', and can be ignored
    if oldTermHmat is not being initialized by this function.
    
    ***TEST***
    
    ***Need to implement for the case of oldTermCurve != None (i.e., we are 
    creating difference lists)***
    """
    
    #loop over molPl and bondPl, calling makeDistortionHmats
    # note that the last distortion place should store the 'universal' matrix,
    # and the first distortion is only needed if bondPl==0
    
    
    #valList=np.zeros(len(mcLists.symmetryValArray[termPl]))
    
    if mcLists.isSingleAtom[termPl]:
        #numpy acceleration is good enough?
        valList = mcLists.symmetryValArray[termPl]*termCurves[termPl][0]
    
    else:
        #now we need njit
        valList = fillValList(mcLists.symmetryValArray[termPl],mcLists.callDist[termPl],mcLists.callFirstLast[termPl],termCurves[termPl])
    
    return valList  # just the list for one termPl!


@njit
def addTermMEs_njit(hmat, valList, oldValList, symmetryIndsArray, matEnds, matEndsPl):
    #nee
    
    while matEnds[0,matEndsPl] >= 0: #continue until a break point
        for pl in range(matEnds[0,matEndsPl],matEnds[1,matEndsPl]):
            hmat[symmetryIndsArray[0,pl],symmetryIndsArray[1,pl]] += valList[pl] - oldValList[pl]
            
        matEndsPl+=1
    
    return matEndsPl + 1  # the value after the break point


def addTermHmatMEs(termPl, hmatList,valList,oldValList,mcLists):
    """Add the MEs from a specific universal model term to the Hamiltonians.
    
    hmatIndices[molPl][-1] is the hmat dimension
    
    Need to create Hmat list items if they are not yet there.
    
    ***TEST***
    ***This function or its sub-calls should be numba-accelerated.  This is not
    being implemnted in the 1st draft, as there are other things to debug***
    """
    
    #**** Need to distinguish isSingleAtom scenarios!!! 
    # Note that MEs are in hmatSubLists[molPl][0] for isSingleAtom==1
    
    matEndsPl=0
    for molPl in range(len(hmatList)):
        for bondPl in range(len(hmatList[molPl])):
            for distPl in range(len(hmatList[molPl][bondPl])):
                    
                newMatEndsPl = addTermMEs_njit(hmatList[molPl][bondPl][distPl],valList[termPl],oldValList[termPl],mcLists.symmetryIndsArray[termPl],mcLists.matEnds[termPl],matEndsPl)
                if not mcLists.isSingleAtom[termPl]:
                    matEndsPl = newMatEndsPl
        
        if mcLists.isSingleAtom[termPl]:
            matEndsPl = newMatEndsPl # for single atoms, there is just one set of MEs per molecule
            

def makeAllHmats(mcLists, termCurves):
    """Takes the current parameters and returns a nested lists containing all
    Hamiltonians + a term-resolved breakdown:
    
    HmatList_term[termInd][molInd][bondInd][distortInd]
    HmatList[molInd][bondInd][distortInd] # full Hamiltonians, summed over termInd
    
    Note that the undistorted matrix is only found in HmatList[termInd][molInd][0][0]
    and HmatList[termInd][molInd][bondInd>0][0] == []
    
    ***TEST!!!***
    """
    

    valList = []
    oldValList = []
    for termPl in range(len(mcLists.isSingleAtom)):
        valList += [[]]
        oldValList += [[]]
        valList[termPl] = makeHmatValList(termPl, mcLists, termCurves)
        oldValList[termPl] = np.zeros(valList[termPl].shape[0])
          
    # Now create an hmat list with appropriate dimensionality:
    hmatList=generate_empty_HmatList(mcLists)
          
    # Now sum over terms to populate the HmatList
    for termPl in range(len(mcLists.isSingleAtom)):
        # print('termPl: ' + str(termPl))
        addTermHmatMEs(termPl, hmatList,valList,oldValList,mcLists)
                       # hmatListDims,mcLists.isSingleAtom[termPl],mcLists.hmatIndex)
    

    
    return hmatList, valList, oldValList

def makeAllHmats_test(mcLists, termCurves, whichTermCurves):
    """Test version of makeAllHmats.  

    Parameters
    ----------
    whichTermCurves : list of int
        Set to 0 for Hmat terms to be excluded, and 1 otherwise. To use all terms, 
        set whichTermCurves=0 
        
        For example, to only use term #10:
        whichTermCurves = np.zeros(21, dtype = np.int32)
        whichTermCurves[10] = 1
    """
    
    

    valList = []
    oldValList = []
    for termPl in range(len(mcLists.isSingleAtom)):
        valList += [[]]
        oldValList += [[]]
        if len(whichTermCurves) > 0: # if not all terms will be included
            valList[termPl] = whichTermCurves[termPl] * makeHmatValList(termPl, mcLists, termCurves)
        else:
            valList[termPl] = makeHmatValList(termPl, mcLists, termCurves)
        oldValList[termPl] = np.zeros(valList[termPl].shape[0])
          
    # Now create an hmat list with appropriate dimensionality:
    hmatList=generate_empty_HmatList(mcLists)
          
    # Now sum over terms to populate the HmatList
    for termPl in range(len(mcLists.isSingleAtom)):
        # print('termPl: ' + str(termPl))
        addTermHmatMEs(termPl, hmatList,valList,oldValList,mcLists)
                       # hmatListDims,mcLists.isSingleAtom[termPl],mcLists.hmatIndex)
        
    
    return hmatList, valList, oldValList





