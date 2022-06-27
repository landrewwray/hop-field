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
    
    """
    
    def __init__(self, *args):
        if len(args)==2:          # if it's a single-atom term
            self.element0=args[0]
            self.element1=None    # just one element!
            self.term=args[1][0]
            self.hop=False        # Not a hopping term
            self.curve=None
            
            self.val=float(self.term[1])
            self.max=float(self.term[3])
            self.min=float(self.term[2])
        else:
            self.element0=args[0] # Acted-on element
            self.element1=args[1] # 2nd element
            self.term=args[2][0]
            self.hop=args[3]      # Is it a hopping term? (True/False)
            
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

    def makeCrossTerms(self,allowedTypes,termFile):
        """***!!! Need to test + avoid creating equivalent terms!  
        Function also requires self.pruneTermsList(), which is unfinished
        
        allowedTypes is a list
        
        theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

        """
        
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
                    #Hopping terms:
                    if pl2>=pl1:  #avoid double counting
                        for nextTerm in hopList:
                            if nextTerm[0]==self.elementList[pl2] or nextTerm[0]=='X':
                                if ((nextTerm[2] in self.orbSyms[pl1]) and (nextTerm[3] in self.orbSyms[pl2])) or ((nextTerm[2] in self.orbSyms[pl2]) and (nextTerm[3] in self.orbSyms[pl1])):
                                    self.termsList += [Hterm(self.elementList[pl1],self.elementList[pl2],[nextTerm],True)]
                    
                    #***Note!  The 'E' Coulomb terms between different atom types are duplicated
        self.pruneTermsList
        
    def pruneTermsList(self):
        # Remove unnecessary terms from self.termsList - there are still 
        pass
    
    def getOrbSymNum(self,elementName,orbName):
        #get the index of an orbital type for a specific element (the 's' and 'p' orbitals of C have indices 0 and 1)
        for elementType in range(len(self.elementList)):
            if self.elementList[elementType] == elementName:
                #now find the orbital
                for orbNum in range(len(self.orbSyms[elementType])):
                    if self.orbSyms[elementType][orbNum][0] == orbName:
                        return orbNum
        
        return None
                

# theUM = UniversalModel(['C','Cinit_0.txt'])
# theUM += UniversalModel(['H','Hinit_0.txt'])  # define __add__ appropriately

# theUM.makeCrossTerms(['all'],'Hop_generic0.txt')


