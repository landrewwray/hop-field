#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022

Develpment notes (Jun 15 2022):
    1. ***The function UniversalModel.pruneTermsList() should be updated to remove 'double counting'
    Coulomb terms. (one extra term for every pair of elements)  Currently the 'E' 2-body Coulomb
    terms are duplicated between pairs of elements.  This means that there are equivalent 
    curves describing the "C-H" interaction and "H-C" interaction. 

    2. It may help to create a single list of all UM terms as well as their min/max values, and
    functions to cross-update between the UniversalModel object and this list.  (Using pointers
    could make this more elegant, but less Pythonic.)

@author: lawray
"""
import universalmodel.sigmoidcurve as sc
import numpy as np

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


def loadTerms(filePath='Hinit_0.txt'):
    """ Read out all space-delineated terms following lines that begin with the '@' symbol,
    and provide output as a list of lists.  Lines with length 0 or begining with '#' are disregarded
    
    theLists = loadTerms('Cinit_0.txt')
    """
    
    
    filePath='AtomInit/'+filePath

    atList=findAt(filePath)
    
    theLists=[]
    for pl in atList:
        theLists+=[[]]
    
    searchfile = open(filePath, 'r')
    atPl=-1 # indicates the parameter type field
    for line in searchfile:
        if '@' in line:
            atPl+=1
        elif atPl>-1:
            if not '#' in line:  # inserting '#' anywhere on a line will comment it out
                if len(line)>1:
                    theLists[atPl]+=[line.split()]
        
    searchfile.close()
    
    return theLists

class Hterm:
    """ Stores the types of of Hamiltonian term in Hterm.termsList.  The model paramterers are stored in
    (for a single number) self.val or (for a SigmoidCurve curve) self.curve 
    
    max and min values are found in (single value) self.max and self.min or
    (SigmoidCurve)  self.curve.Imax, self.curve.Imin
    
    The SigmoidCurve parameter values are stored in self.curve.I_of_r, but one must run
    self.curve.make_sAmps() after changing any value. Additionally, permutations should ideally 
    either act on self.curve.sAmps or act on two adjacent self.curve.I_of_r values simultaneously
    
    self.termType is an integer representing the type of Hamiltonian term referenced:
    0: single atom
    1: Coulomb interaction term
    2: hopping term
    
    """
    
    def __init__(self, *args):        
        self.element0=args[0] # Acted-on element
        
        if len(args)==2:          # if it's a single-atom term
            self.termType=0       # single atom term
            
            self.element1=None    # just one element!
            self.term=args[1][0]
            self.hop=False        # Not a hopping term
            self.curve=None
            
            self.val=float(self.term[1])
            self.max=float(self.term[3])
            self.min=float(self.term[2])
        else:
            self.element1=args[1] # 2nd element
            self.term=args[2][0]
            self.hop=args[3]      # Is it a hopping term? (True/False)
            if self.hop:
                self.termType=2       # hopping term
            else:
                self.termType=1       # Coulomb interaction term
            
            self.curve=sc.SigmoidCurve(int(self.term[-1]),float(self.term[-3]),float(self.term[-2]))
            self.curve.initAmpsLinear(float(self.term[-6]))
            self.curve.init_I_limits([float(self.term[-5]),float(self.term[-4])]) 
                                          # set curve amplitude limits for Monte Carlo
                                          # These are a scalar mutliple of the initial (linear) amplitudes
            
            # equialents of self.val, .max and .min are:
            # self.curve.I_of_r, .Imax, .Imin


class UniversalModel:
    """stores the following variables: elementList, initList, termsList, orbSyms, electronsPerAtom
    """
    
    def __init__(self, *args):
        elementList=['H','Hinit_0.txt'] #default to hydrogen
        
        if len(args)==1:
            elementList=args[0]
        elif len(args)==2:
            self._combineModels(args[0],args[1])
        
        self.maxDist=1e6 # effectively infinite.  A real value is set in self.makeCrossTerms
        
        self.elementList = [elementList[0]] 
        self.initList = [elementList[1]] 
        self._makeAtomTerms() # creates self.termsList --> just orbital energies and SOC
                              # also creates self.orbSyms and self.electronsPerAtom
        
    def __add__(self, other):
        return self._combineModels(other)
    
    def _combineModels(self,other):
        self.elementList+=other.elementList
        self.initList+= other.initList
        self.termsList+=other.termsList
        self.orbSyms+=other.orbSyms
        self.electronsPerAtom+=other.electronsPerAtom
        
        return self
    
    def _makeAtomTerms(self):
        # Only works for a single element (len(elementList)==1)
        # Just creates orbital energies and SOC
        # (start defining the Hamiltonian format here)
        self.termsList = []
        
        theList = loadTerms(self.initList[0])
        
        self.electronsPerAtom=[int(theList[2][0][0])]
        
        theList = theList[0] #just the single atom terms
        
        
        orbSymList=[]
        for nextTerm in theList:
            self.termsList += [Hterm(self.elementList[0],[nextTerm])]
            orbSymList+=[[nextTerm[0], int(nextTerm[4])]]
        self.orbSyms = [orbSymList]
        
        # theDict = []
        # for dictInd in range(self.orbSyms):
        #     theDict+=

    def makeCrossTerms(self,allowedTypes,termFile):
        """***!!! Need to test + avoid creating equivalent terms!  
        Function also requires self.pruneTermsList(), which is unfinished
        
        allowedTypes is a list
        
        theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

        """
        self.maxDist=0
        
        if allowedTypes[0]=='all':
            allowedTypes=set(self.elementList)
        else:
            allowedTypes=set(allowedTypes)
            
        hopList = loadTerms(termFile) # Coulomb terms in theList[0], hopping in theList[1]
        hopList = hopList[0]
        
        for pl1 in range(len(self.elementList)):
            theList = loadTerms(self.initList[pl1]) #Coulomb terms
            theList = theList[1] # interatomic terms only
            
            for pl2 in range(len(self.elementList)):
                if (self.elementList[pl1] in allowedTypes) and (self.elementList[pl2] in allowedTypes):
                    #if both elements are allowed, add the terms!
                    #Coulomb interactions:
                    for nextTerm in theList:
                        if nextTerm[0]==self.elementList[pl2] or nextTerm[0]=='X':
                            self.termsList += [Hterm(self.elementList[pl1],self.elementList[pl2],[nextTerm],False)]
                            
                            if self.termsList[-1].curve.dMax > self.maxDist: #keep track of the maximum interaction distance
                                self.maxDist = self.termsList[-1].curve.dMax
                                
                    #Hopping terms:
                    if pl2>=pl1:  #avoid double counting 
                        for nextTerm in hopList:
                            if nextTerm[0]==self.elementList[pl2] or nextTerm[0]=='X':
                                
                                orbSyms1=np.asarray(self.orbSyms[pl1])[:,0]
                                orbSyms2=np.asarray(self.orbSyms[pl2])[:,0]
                                if ((nextTerm[2] in orbSyms1) and (nextTerm[3] in orbSyms2)) or ((nextTerm[2] in orbSyms2) and (nextTerm[3] in orbSyms1)):
                                    self.termsList += [Hterm(self.elementList[pl1],self.elementList[pl2],[nextTerm],True)]
                                    
                                    if self.termsList[-1].curve.dMax > self.maxDist: #keep track of the maximum interaction distance
                                        self.maxDist = self.termsList[-1].curve.dMax
                    
                    #***Note!  The 'E' Coulomb terms between different atom types are duplicated
        
        self._pruneTermsList() # Remove double-counted 'E' terms
        
        return None
        
        
    def _pruneTermsList(self):
        """Remove unnecessary terms from self.termsList.  This is currently only needed
        for 'E' coulomb terms.  Note that extra E terms can be protected by adding a _keep_ 
        flag in the 2nd term index (e.g.: X _keep_ E -1 -1.5, 0 0.5 10 5])
        """
        
        #1. find "E" terms and make a kill list
        Einds=[]
        for pl in range(len(self.termsList)):
            if self.termsList[pl].term[2] == 'E':
                Einds += [pl]
                
        for pl1 in range(len(Einds)):
            for pl2 in range(pl1+1,len(Einds)):
                # Now, if the elements at Einds(pl2) are reversed:
                if [self.termsList[Einds[pl1]].element0, self.termsList[Einds[pl1]].element1] == [self.termsList[Einds[pl2]].element1, self.termsList[Einds[pl2]].element0]:
                    if self.termsList[Einds[pl2]].term[1] != '_keep_':  # kill the term if it's not labeled _keep_
                        self.termsList.pop(Einds[pl2]) 
                
        return None
    
    def popHs(self, elementName = 'H', orbName = 's'):
        """Eliminates one orbital (defaulting to Hydrogen s) to lift the scalar shift
        degree of freedom for orbital energies.
        """
        for pl in range(len(self.termsList)):
            if self.termsList[pl].termType == 0:
                if (self.termsList[pl].element0 == elementName) and (self.termsList[pl].term[0] == orbName):
                    self.termsList.pop(pl)
                    
                    break
        
    
    def getOrbSymNum(self,elementName,orbName):
        #get the index of an orbital type for a specific element (the 's' and 'p' orbitals of C have indices 0 and 1)
        #this could be faster and more flexible as a dictionary!
        for elementType in range(len(self.elementList)):
            if self.elementList[elementType] == elementName:
                #now find the orbital
                for orbNum in range(len(self.orbSyms[elementType])):
                    if self.orbSyms[elementType][orbNum][0] == orbName:
                        return orbNum
        
        return None
    
    def countElectrons(self, molElementList):
        elNum=0
        
        for elementPl in range(len(self.elementList)):
            elementName = self.elementList[elementPl]
            elNum += np.where(np.asarray(molElementList) == elementName)[0].shape[0] * self.electronsPerAtom[elementPl]
            
        return elNum

# theUM = UniversalModel(['C','Cinit_0.txt'])
# theUM += UniversalModel(['H','Hinit_0.txt'])  # define __add__ appropriately

# theUM.makeCrossTerms(['all'],'Hop_generic0.txt')


