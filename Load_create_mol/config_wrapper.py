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
import re
import copy
import load_create_mol.load_mol as lm
import load_create_mol.distort_mol as dm


def make1MolDistortions(atoms_list, coords_arry, bonds_arry, numDistort, chosenBonds):
    """
    

    Parameters
    ----------
    atoms_list : (LIST) ATOMS IN EACH MOLECULE.
    coords_arry : (ARRAY) COORDINATES OF EACH ATOM
    bonds_arry : (ARRAY) BONDS
    numDistort : (INT) NUMBER OF BONDS TO DISTORT
    chosenBonds : (LIST) BONDS TO DISTORT.

    Returns
    -------
    distortList : (LIST) CONTAINS ATOMS LIST, BONDS THAT WERE DISTORTED AND ALL THE DISTORTIONS.

    """
    buckleDist = 0.02  # angstroms
    stretchDist = 0.01

    numToDistort = min([numDistort, bonds_arry.shape[0]])

    distortList = [[atoms_list, chosenBonds]]

    for bond in chosenBonds:

        chosenAxis = np.random.rand(3)  # chosenAxis will be different for each bond
        newCoordsBucklePlus = dm.distort_molecule(
            coords_arry, bonds_arry, bond, buckleDist, chosenAxis
        )
        newCoordsBuckleMinus = dm.distort_molecule(
            coords_arry, bonds_arry, bond, buckleDist * -1, chosenAxis
        )
        newCoordsStretchPlus = dm.distort_molecule(
            coords_arry, bonds_arry, bond, stretchDist
        )
        newCoordsStretchMinus = dm.distort_molecule(
            coords_arry, bonds_arry, bond, stretchDist * -1
        )
        distortList += [
            [
                newCoordsBucklePlus,
                newCoordsBuckleMinus,
                newCoordsStretchPlus,
                newCoordsStretchMinus,
            ]
        ]

    return distortList


def molNumber(path):
    """
    
    totalMoleculesInList = molNumber(path)

    Parameters
    ----------
    folder : PATH TO FOLDER WITH ALL MOLECULES.

    Returns
    -------
    TYPE
        NUMBER OF MOLECULES IN DIRECTORY.

    """
    return len(lm.loadMol(path))


def allMolDistortions(path):
    """
    

    Parameters
    ----------
    path : PATH TO FOLDER WITH ALL MOLECULES

    Returns
    -------
    distortionLists : MAX SIZE LIST WITH ALL CONFIGS.

    """

    atomsLists, bondsArrays, coordsArrays = lm.loadMol(path)
    
    distortionLists = []

    for index in range(len(atomsLists)):
        numDistort = (bondsArrays[index].shape[0])//2
        chosenBonds = random.sample(
            list(range(bondsArrays[index].shape[0])), numDistort
        )
        distortionLists += [
            [
                make1MolDistortions(
                    atomsLists[index],
                    coordsArrays[index],
                    bondsArrays[index],
                    numDistort,
                    chosenBonds,
                )
            ]
        ]

    return distortionLists

def elementsLists (atomsLists): 
    elementsLists = []
    for molecule in atomsLists:
        elemList = []
        for atom in molecule:
            elemList += [re.findall('([a-zA-Z ]*)\d*.*', atom)[0]]
        elementsLists +=[[elemList]]      
        elemList = []
    return elementsLists    

class ConfigWrapper:
    """
    A wrapper storing configurations ... describe structure
   
   
    A sample call would go here, or possibly a sample class init
    """

    def __init__(self, distortLists=[], atomsLists=[], elementsLists=[], bondsArrays=[], coordsArrays=[]):
        self.distortLists = distortLists
        self.atomsLists = atomsLists
        self.elementsLists = elementsLists
        self.bondsArrays = bondsArrays
        self.coordsArrays = coordsArrays


# def mol_num(self, atomIndex):
