#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:05:08 2022

Development notes:
    
    Debugging:
        1. Construct and test the Hamiltonian!
             - May need to double-check the independence of Hamiltonian term data
             storage in the model creation. Is anything being overwritten?


@author: lawray
"""

import copy
import numpy as np
from numba import njit, int32, float64
from numba.experimental import jitclass

# import monte_carlo.sigmoidcurve_numba as scn
import monte_carlo.decant_terms as dec

# mcLists = MCwrapper()
# mcLists.callDist = callDist  # callDist[termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
# mcLists.firstInd = firstInd  # [termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
# mcLists.lastInd = lastInd    # [termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
# mcLists.symmetryRowArray = symmetryRowArray  # [termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
# mcLists.symmetryColArray = symmetryColArray  # [termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
# mcLists.symmetryValArray = symmetryValArray  # [termNum][molPl][distortedBondNum][distortionNum] gives a numpy array

# mcLists.isSingleAtom = isSingleAtom           # isSingleAtom[termNum] is boolean


# spec = [
#     ('isSingleAtom', int32),            
#     ('dMin', float64),          
#     ('dMax', float64),
#     ('I_of_r', float64[:]),
#     ('sAmps', float64[:]),
#     ('_sWidth', float64),
#     ('_sStart', float64[:]),
#     ('Ix', float64[:]),
#     ('Imax', float64[:]),
#     ('Imin', float64[:]),
# ]

# @jitclass(spec)
# class stateWrapper:
    
#     def __init__(self):
#         self.isSingleAtom=

# @jit
# def makeCommonHmat_np(symmetryRowArray,symmetryColArray,symmetryValArray,theCurve,callDist,firstIndlastInd)
#     pass

# def makeCommonHmat(termPl,molPl,mcLists):
#     # pass the relevant info to makeCommonMat_np
#     pass
  
spec1Term = [
    ('inds', int32[:,:]),            
    ('vals', float64[:]),          
]
  
@jitclass(spec1Term)
class Term1DistWrapper(object):
    """ Stores a list of Hamiltionian terms.
    
    self.inds contains row/col inds in [row0_col1] format

    Note that the data types defined here will need to be updated if floatdt or
    intdt are updated in hmat_stitcher.make_monte_carlo_lists !!!
    """
    
    def __init__(self, valArray, indsArray):
        self.vals=valArray
        self.inds=indsArray
    
@njit
def makeEmptyTermWrapper():
    theWrapper = Term1DistWrapper(np.empty((0),dtype=np.float64), np.empty((2,0),dtype=np.int32))

    return theWrapper

@njit
def makeDistTermHmats(distMats,callDistList, callFirstLastList, symValArrayList, symmetryIndsArray,termCurve, createUndistorted):
    """
    Create the term-resolved 'distortion' Hamiltonian inside Term1DistWrapper
    jitclass objects.  Note that this creation involves coppying symmetryIndsArray,
    which should be unnecessary. 
    
    ***Also, the values should be the different between current and previous values --
    how to implement this for compatibility with njit?
    
    ***Test!!!

    """
    
    
    # identify the range of distortion scenarios to consider
    if createUndistorted:
        distRange=range(len(callDistList))
    else:
        distRange=range(1,len(callDistList))
        
    for distPl in distRange:
        
        # first initialize the Term1DistWrapper with symmetry values
        # *** There should be an option to only init the values, not the inds!
        
        distMats[distPl].vals=symValArrayList[distPl]
        distMats[distPl].inds=symmetryIndsArray[distPl]
        
        # = Term1DistWrapper(symValArrayList[distPl], symmetryIndsArray[distPl])
        
        for callPl in range(len(callDistList[distPl])):  
            #distMats[distPl].vals[callFirstLastList[distPl][0,callDistList[distPl][callPl]]:callFirstLastList[distPl][1,callDistList[distPl][callPl]]] *= termCurve.readVal(callDistList[distPl][callDistList[distPl][callPl]])
            distMats[distPl].vals[callFirstLastList[distPl][0,callPl]:callFirstLastList[distPl][1,callPl]] *= termCurve.readVal(callDistList[distPl][callPl])  

    return None
    
def initEmptyMatsList(matsList,callDistList):
    
    for distPl in range(len(callDistList)):
        matsList+=[makeEmptyTermWrapper()]
        
    return None
        
def makeDistortionHmats(matsList,mcLists,mi,oldTermCurve,termCurve,isSingleAtom):
    """Create Hmats for all distortions of a single bond.  This function unpacks
    the needed variables and calls njit-accelerated functions.
    
    if isSingleAtom=0 and mi[2]==0, then update the undistorted matrix (distiortionPl==0)
    mi = [termPl, molPl, bondPl]
    Note that these 'matrices' are just index lists!
    """
    

    if isSingleAtom:  #*** check '* termCurve' -- is it just a number?
        if oldTermCurve == None:
            oldTermCurve = [0]
        
        #***** this will probably overwrite and lose the pointer!
        print('try')
        matsList += [Term1DistWrapper(mcLists.symmetryValArray[mi[0]][mi[1]] * (termCurve[0]-oldTermCurve[0]),mcLists.symmetryIndsArray[mi[0]][mi[1]])]
        print('done')
        
        
    else:
        # createUndistorted=False
        # if mi[2] == 0: createUndistorted = True
        createUndistorted = True  # create all matrices in the initial implementation, for simplicity
        
        if oldTermCurve != None:
            theTermCurve = termCurve-oldTermCurve # make sure termcurve is not updated by this!
        else:
            theTermCurve = termCurve
                    
        if len(matsList) == 0:
            initEmptyMatsList(matsList,mcLists.callDist[mi[0]][mi[1]][mi[2]])
            
        makeDistTermHmats(matsList,mcLists.callDist[mi[0]][mi[1]][mi[2]], mcLists.callFirstLast[mi[0]][mi[1]][mi[2]], mcLists.symmetryValArray[mi[0]][mi[1]][mi[2]], mcLists.symmetryIndsArray[mi[0]][mi[1]][mi[2]],theTermCurve, createUndistorted)
    
        return None

"""
        mcLists.callDist       # callDist[termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
        mcLists.callFirstLast  # [termNum][molPl][distortedBondNum][distortionNum] --> [first0_last1,callNumber] 
                                                                                          # 2D numpy array

        mcLists.symmetryValArray 
        mcLists.symmetryRowArray   # [termNum][molPl][distortedBondNum][distortionNum] 
                                   # 2D numpy array of [row0_col1,indNumber]

        mcLists.isSingleAtom       # isSingleAtom[termNum] is boolean
        
        When isSingleAtom==True, the 'call...' lists are omitted and there are no 
        [distortedBondNum][distortionNum] indices
    """


def updateTermHmats(termPl, mcLists, termCurve, isSingleAtom, oldTermCurve=None, oldTermHmat=None):
    """
    
    Set oldTermCurve=-1 to assume that previous values were 0, and set 
    oldTermHmat=[] if it has not yet been initialized
    
    Note that the returned list is the same object as 'oldTermHmat', and can be ignored
    if oldTermHmat is not being initialized by this function.
    
    ***TEST***
    
    ***Need to implement for the case of oldTermCurve != None (i.e., we are 
    creating difference lists)***
    """
    
    #loop over molPl and bondPl, calling makeDistortionHmats
    # note that the last distortion place should store the 'universal' matrix,
    # and the first distortion is only needed if bondPl==0
    
    
    newInit=False
    if (oldTermHmat == None) or (oldTermCurve == None):
        newInit=True
    
    if oldTermHmat == None:
        termHmats = []
    else:
        termHmats = oldTermHmat        
            
    for molPl in range(len(mcLists.callDist[termPl])):
        if newInit: termHmats += [[]]    
        
        if isSingleAtom:
            # if newInit: termHmats[molPl] += [[]]
            mi = [termPl, molPl]
            
            makeDistortionHmats(termHmats[molPl],mcLists,mi,oldTermCurve,termCurve,isSingleAtom)
            
        else:
            for bondPl in range(len(mcLists.callDist[termPl][molPl])):
                if newInit: termHmats[molPl] += [[]]
                
                mi=[termPl,molPl,bondPl]
                makeDistortionHmats(termHmats[molPl][bondPl],mcLists,mi,oldTermCurve,termCurve,isSingleAtom)
    
    return termHmats
    
    #term-resolved Hmats should be lists of matrix elements, not full numpy arrays
    


def readMaxDims(HmatList_term,isSingleAtom):
    """nested list of [molPl][bondPl][numDist]
    
    ***Test!!!***
    """
    
    
    nonSAterm=np.where(isSingleAtom==0)[0][0]
    
    hmatDims = []
    for molPl in range(len(HmatList_term[nonSAterm])):
        hmatDims += [[]]
        for bondPl in range(len(HmatList_term[nonSAterm][molPl])):
            hmatDims[molPl] += [[]]
            hmatDims[molPl][bondPl] = len(HmatList_term[nonSAterm][molPl][bondPl])
    
    return hmatDims

def initHmats(hmatList, hmatListDims, hmatIndices):
    for molPl in range(len(hmatListDims)):
        if len(hmatList)-1 < molPl: # if we're creating the Hmats for the first time
            hmatList+= [[]]
            
        for bandPl in range(len(hmatListDims[molPl])):
            if len(hmatList[molPl])-1 < bandPl: # if we're creating the Hmats for the first time
                hmatList[molPl]+= [[]]
                
            for distortPl in range(hmatListDims[molPl][bandPl]-1):
                if len(hmatList[molPl][bandPl])-1 < distortPl:
                    hmatList[molPl][bandPl] += [[]]
                    hmatList[molPl][bandPl][distortPl] = np.zeros((hmatIndices[molPl][-1],hmatIndices[molPl][-1]))
    

def addTermHmatMEs(hmatList, hmatSubLists, hmatListDims, isSingleAtom,hmatIndices):
    """Add the MEs from a specific universal model term to the Hamiltonians.
    
    hmatIndices[molPl][-1] is the hmat dimension
    
    Need to create Hmat list items if they are not yet there.
    
    ***TEST***
    ***This function or its sub-calls should be numba-accelerated.  This is not
    being implemnted in the 1st draft, as there are other things to debug***
    """
    
    #**** Need to distinguish isSingleAtom scenarios!!! 
    # Note that MEs are in hmatSubLists[molPl][0] for isSingleAtom==1
    
    
    if hmatList == []:  #init the hamiltonians as empty numpy arrays, if not previously done
        initHmats(hmatList, hmatListDims, hmatIndices)
        

    for molPl in range(len(hmatListDims)):
            
        for bandPl in range(len(hmatListDims[molPl])):
                
            lastDistort=hmatListDims[molPl][bandPl]-1 # address of the distortion-invariant matrix elements
            for distortPl in range(lastDistort): # 
                if not isSingleAtom:
                    # print('!!!for future updates, need to deal with possible leading [] when bandPl != 0 ')
                    # (check the definition of createUndistorted above)
                    for pl in range(len(hmatSubLists[molPl][bandPl][distortPl].vals)):
                        hmatList[molPl][bandPl][distortPl][hmatSubLists[molPl][bandPl][distortPl].inds[0,pl],hmatSubLists[molPl][bandPl][distortPl].inds[1,pl]] += hmatSubLists[molPl][bandPl][distortPl].vals[pl]
                    
                    # for the last (universal MEs) distortion index in hmatSubLists, add to all lists:
                    for pl in range(len(hmatSubLists[molPl][bandPl][lastDistort].vals)):
                        hmatList[molPl][bandPl][distortPl][hmatSubLists[molPl][bandPl][lastDistort].inds[0,pl],hmatSubLists[molPl][bandPl][lastDistort].inds[1,pl]] += hmatSubLists[molPl][bandPl][lastDistort].vals[pl]
                else:
                    for pl in range(len(hmatSubLists[molPl][0].vals)):
                        hmatList[molPl][bandPl][distortPl][hmatSubLists[molPl][0].inds[0,pl],hmatSubLists[molPl][0].inds[1,pl]] += hmatSubLists[molPl][0].vals[pl]
                    
        

def makeAllHmats(mcLists, termCurves):
    """Takes the current parameters and returns a nested lists containing all
    Hamiltonians + a term-resolved breakdown:
    
    HmatList_term[termInd][molInd][bondInd][distortInd]
    HmatList[molInd][bondInd][distortInd] # full Hamiltonians, summed over termInd
    
    Note that the undistorted matrix is only found in HmatList[termInd][molInd][0][0]
    and HmatList[termInd][molInd][bondInd>0][0] == []
    
    ***TEST!!!***
    """
    

    hmatList_term =[]
    for termPl in range(len(mcLists.isSingleAtom)):
        hmatList_term += [[]]
        hmatList_term[termPl] = updateTermHmats(termPl, mcLists, termCurves[termPl], mcLists.isSingleAtom[termPl])
        
          #updateTermHmats(mcLists,termCurves[termPl],      -1,     mcLists.isSingleAtom[termPl])
          #               mcLists, current term params, old params,   is it a single atom term?
          # return the difference between new params and old params

          #problems: 
              # no fixed type for old_params and termCurves[termPl]
              # --> 
          
    # Now identify the dimensionality
    
    hmatListDims=readMaxDims(hmatList_term, mcLists.isSingleAtom)
    
          
    # Now sum over terms to get the HmatList
    hmatList = []
    for termPl in range(len(mcLists.isSingleAtom)):
        hmatList += [[]]
        # print('termPl: ' + str(termPl))
        addTermHmatMEs(hmatList[termPl],hmatList_term[termPl],hmatListDims,mcLists.isSingleAtom[termPl],mcLists.hmatIndex)
            # *** Implement!  Use HmatList to get dimensions once it's implemented; otherwise pass hmatDims
    
    return hmatList, hmatList_term


# def generateTval(termCurves, termNums, mcLists, steps=20):
#     """***Not yet implemented***
    
#     """

    
    
    
#     #finish by reloading the original parameters:
#     termCurves = dec.decant_UM_terms(mcLists.theUM) # numba SigmoidCurve curves and single atom values
#     termNums = dec.readTerms(termCurves) # just the underlying parameters
    
#     return Tval
    
# def run_mc(termCurves, termNums, mcLists, Tval, samplesPerTrial, steps):
    
    
#     #return X, y, goodMoveRate 
    