# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:12:03 2022

@author: danie
"""

import numpy as np
import pandas as pd
from math import floor
from os import listdir
import copy
import glob
# import tightBinding as tbi
import load_create_mol.config_wrapper as wrapper
    
    
######################################################
###### molecular data loading functions #####
def findAt(filePath):
    # find @ symbols
    # atList=findAt('Cyclopropane(C3H6).mol2')
    searchfile = open(filePath, 'r')
    linePl=0
    atList=[]
    for line in searchfile:
        if '@' in line: 
            # print(line)
            atList+=[linePl]

        linePl+=1
        
    searchfile.close()
    # print(atList)
    return atList
   
def loadFile(filePath):

    # atoms_list, coords_arry, bonds_arry = loadFile('Cyclopropane(C3H6).mol2')
    atList=findAt(filePath)
    the_coords=pd.read_csv(filePath,sep='\s+',nrows=atList[2]-atList[1]-1, skiprows=atList[1]+1, header=None)
    the_bonds=pd.read_csv(filePath,sep='\s+',nrows=atList[3]-atList[2]-1, skiprows=atList[2]+1, header=None)
    
    atoms_list=the_coords.loc[:,1].to_list()
    coords_arry=the_coords.loc[:,2:4].to_numpy()
    bonds_arry=the_bonds.to_numpy()
    bonds_arry[:,:3]=bonds_arry[:,:3]-1
    
    return atoms_list, coords_arry, bonds_arry

def loadMol(folder, sizeCap=-1, moleculesToLoad = []):
    # atomsLists, bondsArrays, coordsArrays = loadMol("Molecular Structure Data/*.mol2")
    
    nameGlob=glob.glob(folder)
    if not len(nameGlob):
        print('Error!  No molecule files found in path: ' + folder)
    
    nameGlob.sort()    # alphabetical order
    
    moleculesList = []
    listPl=0
    for filename in nameGlob:
        #moleculesList += [loadFile(filename)] 
        
        if len(moleculesToLoad)>0:  
            if len(moleculesList) >= len(moleculesToLoad):
                break
            elif moleculesToLoad[listPl] == 1:
                moleculesList += [loadFile(filename)] 
                
            listPl += 1
        else: # len(moleculesToLoad) == 0 will load all molecules
            moleculesList += [loadFile(filename)] 

    atomsLists = [molecule[0] for molecule in moleculesList]
    coordsArrays = [molecule[1] for molecule in moleculesList]
    bondsArrays = [molecule[2] for molecule in moleculesList]
    
    print(sizeCap)
    if sizeCap > 0: # if there is a cap on the number of atoms, then remove
        pl=0        # molecules that exceed the cap:
        while pl<len(atomsLists):
            if len(atomsLists[pl]) > sizeCap:
                atomsLists.pop(pl)
                coordsArrays.pop(pl)
                bondsArrays.pop(pl)
            else:
                pl+=1
                
    return atomsLists, bondsArrays, coordsArrays

def load_struct_info(folder, bondsPerMol, sizeCap = -1, moleculesToLoad = []):
    
    atomsLists, bondsArrays, coordsArrays = loadMol(folder,sizeCap,moleculesToLoad)
    distortLists, molNumList = wrapper.allMolDistortions(folder,bondsPerMol,atomsLists, bondsArrays, coordsArrays)
    elementsLists = wrapper.elementsLists(atomsLists)
    
    structInfo = wrapper.ConfigWrapper(distortLists, atomsLists, elementsLists, bondsArrays, coordsArrays) # distortLists, atomsLists, elementsLists, bondsArrays, coordsArrays
    return structInfo
    
