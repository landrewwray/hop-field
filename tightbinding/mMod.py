#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:21:49 2022

@author: lawray
"""

import numpy as np
import pandas as pd
from math import floor
from os import listdir
import copy
import tightBinding as tbi



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
            
######################################################
##### defining the Hamiltonian ######


    
    
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
    
class oneMol:
    """Structure to contain parameters for a single molecule
    -the structure and atom types
    -the Hamiltonian
    -Hamiltonian sub-represenations
    
    :iVar myName: name of the molecule
    :iVar atomicOrbBases: Atomic orbital bases for a tight binding model
    :iVar atom_names: list of elements present (e.g. 'C', 'N', 'Au')
    :iVar Hmat: The Hamiltonian, excluding density-dependent terms that may
                be added later
    
    """
    
    def __init__(self,pathStem=[], molName='Cyclopropane(C3H6).mol2'):
        #key 
        
        
        if pathStem==[]:
            pathStem="Molecular Structure Data/"
        
        self.myName=molName
        self.atoms_list, self.coords_arry, self.bonds_arry = loadFile(pathStem+molName)
        
        # self.subReps=() # this will be a tuple of numpy arrays, 
        #                 # initialized by a moleculeArchive
        
        # specify the recognized element+orbital combinations
        self.atomicOrbBases={'C': [['s', 'p'],[1, 3]], 'H': [['s'],[1]]}
        
        self.atom_names=[] #list standard element abbreviations (e.g. C, Au)
        for theAtom in range(len(self.atoms_list)):
            lenChk=self.atoms_list[theAtom][1]
            if lenChk.isnumeric(): #if the element name is one letter
                self.atom_names+=[self.atoms_list[theAtom][0]]
            else:
                self.atom_names+=[self.atoms_list[theAtom][:2]]

        # Now define the tight binding Hamiltonian basis (gives state2num)
        basisPl=0                
        self.atomBasisLocs=[]
        atomNum=0
        for theAtom in self.atom_names:
            try:
                numManifolds=len(self.atomicOrbBases[theAtom][0])
            except:
                print('Atom number '+str(atomNum)+' is unrecognized in '+self.myName)
                numManifolds=0
            if numManifolds>0:
                self.atomBasisLocs+=[copy.deepcopy(self.atomicOrbBases[theAtom])]
                for pl in range(numManifolds):
                    self.atomBasisLocs[atomNum][1][pl]=basisPl
                    basisPl+=self.atomicOrbBases[theAtom][1][pl]
                self.atomBasisLocs[atomNum]=dict(np.transpose(self.atomBasisLocs[atomNum])) #dictionary instead!
                                                                #an orbital can be killed for 1 atom by setting a negative index:
                                                                #self.atomBasisLocs[atomNum]['s']=-1
                # now convert back from char to int for basis locations (format changed due to np.transpose list-->array)
                for orbLetter in self.atomicOrbBases[theAtom][0]:
                    self.atomBasisLocs[atomNum][orbLetter]= int(self.atomBasisLocs[atomNum][orbLetter])
                
            else:
                self.atomBasisLocs+=[]
            atomNum+=1
        self.totalBasisSize=basisPl # size of the Hamiltonian
        self.Hmat=np.zeros((self.totalBasisSize,self.totalBasisSize))
        
        # Now create the num2state list
        num2state=[]
        atomNum=0
        for theAtom in self.atom_names:
            numOrbs=len(self.atomicOrbBases[theAtom][0])
            for orbPl in range(numOrbs):
                num2state+=tbi.listOrbBasis(atomNum, self.atomicOrbBases[theAtom][0][orbPl])
                
            atomNum+=1
        self.num2state=num2state
        
    def findPairs(self, maxDist):
        self.pairs_list=[]
        for ii in range(len(self.atoms_list)):
            for jj in range(ii+1,len(self.atoms_list)):
                theDist=np.linalg.norm(self.coords_arry[jj,:]-self.coords_arry[ii,:])
                if theDist <= maxDist:
                    self.pairs_list+=[[ii,jj,theDist]]
                
                    
class moleculeArchive:
    # Class storing all molecule configurations and stem Hamiltonians,
    # and defining the Hamiltonian terms

    def __init__(self,pathStem=[], moleculeNames=[]):
        # init self.molNames, self.theMolecules
        
        #define a default data folder:
        if pathStem==[]:
            pathStem="Molecular Structure Data/"

        #if molecules are not specified, read all .mol2 data in the folder
        if moleculeNames==[]:
            fileList=listdir(pathStem)
            
            for fileName in fileList:
                if fileName[-5:]=='.mol2':
                    moleculeNames=moleculeNames+[fileName]

        self.molNames=tuple(moleculeNames)
        
        #now init all of the single molecules
        self.theMolecules=[]
        print(self.molNames)
        for molName in self.molNames:
            self.theMolecules=self.theMolecules + [oneMol(pathStem,molName)]

    def defineHmodel(self):
        #define the terms of the model
        #this should probably come from a separate file!
        

        
        #*** can I enter text in a separate file, like a matlab script?
        pass
        
        
    def findAtomPairs(self,molNum,dMax):
        #create a list of all pairs of atoms separated by less than dMax
        pass
    
    def makeDeformation(self,theBond,defType):
        # defines a single deformation, and makes sure it works
        pass
    
    def makeDeformatonList(self):
        # creates a full list of reasonable deformations
        pass
    
    
    
    
    
    def makePairwiseHmat(self,molNum,pairNum,Hterms=[]):
        #applies non-correlated Hmat terms
        pass
    
    def rotateHmat(self,molNum,pairNum):
        # change the basis from psigma=px to the bond orientation
        pass
    
    def createHmatSubReps(self):
        # create the SubRep Hamiltonians 
        # (for everything but correlated terms)
        pass
    
    # still need to define the framework for correlated Hamiltonian terms
    # rougly speaking: need to define an annihilation operator 

    
    
            