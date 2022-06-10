# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:04:43 2022

@author: danie
"""

import numpy as np
import random
import pandas as pd
from math import floor
from os import listdir
import copy
import Load_mol as lm
import distort_mol as dm

# import tightBinding as tbi





atoms_list, coords_arry, bonds_arry = lm.loadFile('"C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/(5)Helicene(C22H14).mol2"')


def make1MolDistortions(coords_arry, bonds_arry, numDistort):
    buckleDist = 0.02  # angstroms
    stretchDist = 0.01
   
    numToDistort=min([numDistort//2,bonds_arry.shape[0])
    chosenBonds = random.sample(list(range(bonds_arry.shape[0])),numToDistort)
    chosenAxis = np.random.rand(3)
                               
    for bond in chosenBonds:
        newCoordsBucklePlus= distort_molecule(coords_arry,bonds_arry,bond,fixPosSet=set(),buckleDist,chosenAxis)
        coords_arry = newCoordsBucklePlus
        
    for bond in chosenBonds:
        newCoordsBuckleMinus= distort_molecule(coords_arry,bonds_arry,bond,fixPosSet=set(),buckleDist*-1,chosenAxis)
        coords_arry = newCoordsBuckleMinus
        
    for bond in chosenBonds:
        newCoordsStretchPlus= distort_molecule(coords_arry,bonds_arry,bond,fixPosSet=set(),stretchDist,[])
        coords_arry = newCoordsStretchPlus
        
    for bond in chosenBonds:
        newCoordsStretchMinus= distort_molecule(coords_arry,bonds_arry,bond,fixPosSet=set(),stretchDist*-1,[])
        coords_arry = newCoordsStretchMinus
        
    stretchList = [newCoordsStretchPlus, newCoordsStretchMinus]
    buckleList = [newCoordsBucklePlus, newCoordsBuckleMinus]
    
    distortList = [atoms_list, stretchList, buckleList]
    
    return distortList
    
    #when buckling, buckle in positive and negative directions
    #distortList=[atoms_list]
    #distortList+=[distort_molecule(coords_arry,bonds_arry,bondNum,fixPosSet=set(),distortDist=buckleDist,distortAxis=np.random.rand(3))]
    #distortList+=[distort_molecule(coords_arry,bonds_arry,bondNum,fixPosSet=set(),distortDist=buckleDist*-1,distortAxis=np.random.rand(3))]  

def molNumber(folder):
    """
    
    totalMoleculesInList = molNumber(path)

    Parameters
    ----------
    folder : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return len(lm.loadMol(folder))
   
def electroNumber (atoms):
    # should count electrons per atom in atom list- should we make a class for atoms and electrons?

class ConfigWrapper:
    """
    A wrapper storing configurations ... describe structure
   
   
    A sample call would go here, or possibly a sample class init
    """

    def __init__(self, distortLists = [], moleculeNumbers, electronNumbers):
        self.configList= distortLists
        # self.configList+= [distortList]
        # distortList contains 3 arrays, each with a set of coordinates
        # len(self.configList) is the number of deformations applied on all molecules
         
        self.moleculeNum= moleculeNumbers
        self.electronNum=[]
    def 