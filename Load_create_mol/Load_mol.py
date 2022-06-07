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
## import tightBinding as tbi

    
    
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

def loadMol(folder):
    # moleculesList = loadMol("C:/Users/username/*/hop-field/Molecular Structure Data/*.mol2")
    return([loadFile(filename) for filename in glob.glob(folder)])