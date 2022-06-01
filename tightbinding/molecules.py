#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:50:12 2022

Defines Molecule and MoleculeList classes to store single molecules and 

@author: L. Andrew Wray
"""

import numpy as np
import pandas as pd
from math import floor
from os import listdir
import copy

######################################################
##### bond distortion functions ######

def distort_bond(theCoords,_theBonds_,bondNum,distortDist=0.01,distortAxis=[]):
    """*** tested, but buckling distortions not well examined
    
    Shift the 2nd atom listed in a given bond, and return both the new coordinate and the distortion vector
    buckling distortions are orthogonalized from the bond axis.
    
    >>>newCoords, distortVector = distort_bond(coords_arry,bonds_arry,1) 
    """
    
    _theCoords_=copy.deepcopy(theCoords) #modified data will be returned in newCoords
    if distortAxis==[]:
        distortAxis=bondNum
    
    isBuckle=0
    chosenBondVector=_theCoords_[_theBonds_[bondNum,2],:]-_theCoords_[_theBonds_[bondNum,1],:]
    #if distortAxis is an integer, indicating a bond axis
    if type(distortAxis) == int:
        distortVector=_theCoords_[_theBonds_[distortAxis,2],:]-_theCoords_[_theBonds_[distortAxis,1],:] 
        #if it's not ==bondNum, then assume it's a buckling distortion 
        if distortAxis != bondNum:
            isBuckle=1    
    else:  # if it's a vector
        isBuckle=1
        distortVector=distortAxis
        print("got here!2")
        
    if isBuckle:
        distortVector=np.cross(np.cross(chosenBondVector,distortVector),chosenBondVector)
        
    #now set the length
    distortVector=distortDist*distortVector/np.linalg.norm(distortVector)
    newCoords=_theCoords_
    newCoords[_theBonds_[bondNum,2],:]+= distortVector
    
    return newCoords, distortVector
    
def distort_molecule(theCoords,theBonds,bondNum,fixPosSet=set(),distortDist=0.01,distortAxis=[]):
    """*** tested for several scenarios
    
    Distort the entire structure by stretching just one bond.
    If other bonds cannot be held constant, (say, in a benzene ring)
    then enter the atoms to hold fixed (stretching bonds) in fixPosList
    """
    
    #first find the distortion direction, and distort the initial bond by moving bonds_arry[bondNum,2]
    newCoords, distortVector = distort_bond(theCoords,theBonds,bondNum,distortDist,distortAxis)
    
    # do not allow atoms on the original bond to move again:
    fixPosSet=set(fixPosSet) #allow lists to be entered
    fixPosSet=fixPosSet.union({theBonds[bondNum,1], theBonds[bondNum,2]})
    
    #start from the moved atom, and look for atoms it connects to
    startingPointSet={theBonds[bondNum,2]}
    while len(startingPointSet)>0:
    
        #1. find all connected atoms
        connectedPoints=set()
        for startingPoint in startingPointSet:  
            connectedPoints=connectedPoints.union(theBonds[np.where(theBonds[:,1]==startingPoint),2].flatten())
            connectedPoints=connectedPoints.union(theBonds[np.where(theBonds[:,2]==startingPoint),1].flatten())
        connectedPoints=connectedPoints-fixPosSet
        
        #2. now move all the connectedPoints, make sure they won't move again, and set them as the new startingPointList
        newCoords[list(connectedPoints),:]+=distortVector
        fixPosSet=fixPosSet.union(connectedPoints)
        startingPointSet=copy.copy(connectedPoints)
    
    return newCoords