#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022

@author: lawray
"""
#import sigmoidcurve as sc


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
    def __init__(self, *args):
        if len(args)==2:          # if it's a single-atom term
            self.element0=args[0]
            self.element1=None    # just one element!
            self.term=args[1][0]
            self.hop=False        # Not a hopping term
        else:
            self.element0=args[0] # Acted-on element
            self.element1=args[1] # 2nd element
            self.term=args[2]
            self.hop=args[3]      # Is it a hopping term? (True/False)

class UniversalModel:
    def __init__(self, *args):
        elementList=['H','Hinit_0.txt'] #default to hydrogen
        
        if len(args)==1:
            elementList=args[0]
        elif len(args)==2:
            self._combineModels(args[0],args[1])
        
        self.elementList = [elementList[0]] 
        self.initList = [elementList[1]] 
        self._makeAtomTerms() # creates self.termsList --> just orbital energies and SOC
        
        
    def __add__(self, other):
        return self._combineModels(other)
    
    def _combineModels(self,other):
        self.elementList+=other.elementList
        self.initList+= other.initList
        self.termsList+=other.termsList
        
        return self
    
    def _makeAtomTerms(self):
        # Only works for a single element (len(elementList)==1)
        # Just creates orbital energies and SOC
        # (start defining the Hamiltonian format here)
        self.termsList = []
        
        theList = loadTerms(self.initList[0])
        theList = theList[0] #just the single atom terms
        
        for nextTerm in theList:
            self.termsList += [Hterm(self.elementList[0],[nextTerm])]

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
            
            for pl2 in range(pl1,len(self.elementList)):
                if (self.elementList[pl1] in allowedTypes) and (self.elementList[pl2] in allowedTypes):
                    #if both elements are allowed, add the terms!
                    #Coulomb interactions:
                    for nextTerm in theList:
                        if nextTerm[0]==self.elementList[pl2] or nextTerm[0]=='X':
                            self.termsList += [Hterm(self.elementList[pl1],self.elementList[pl2],[nextTerm],False)]
                    #Hopping terms:
                    for nextTerm in hopList:
                        if nextTerm[0]==self.elementList[pl2] or nextTerm[0]=='X':
                            self.termsList += [Hterm(self.elementList[pl1],self.elementList[pl2],[nextTerm],True)]
                    
                    #***Note!  This will create unnecessary terms such as p-orbital hopping between Hydrongen atoms
                    #***also, the Coulomb terms are all defined from the 1st encountered element's interaction list
        self.pruneTermsList
        
    def pruneTermsList(self):
        # Remove unnecessary terms from self.termsList, such as p-orbital hopping between Hydrongen atoms
        pass
    

# theUM = UniversalModel(['C','Cinit_0.txt'])
# theUM += UniversalModel(['H','Hinit_0.txt'])  # define __add__ appropriately

# theUM.makeCrossTerms(['all'],'Hop_generic0.txt')


