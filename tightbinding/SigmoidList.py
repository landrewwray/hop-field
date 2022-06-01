#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 00:07:18 2022

Defines 2-body tight binding field terms and a universalModel class in  which
they can be used to generate molecular Hamiltonians.

import tightBinding as tbi
import mMod   #molecule loding library

theUModel = tbi.universalModel()  # create a universal model with the field 
                                  # terms and limits set in tightBinding.py

myMolecules = mMod.moleculeArchive() # load molecule data

theUModel.createTBHamiltonians(theMolecules) # create Hamiltonians (no 
                    # charge-charge interactions) Hamiltonians are stored in 
                    # myMolecules.theMolecules[molNum].theHopHmat

theUModel.evaluateTB_GS(theMolecules) # find eigenstates and energies
        # myMolecules.theMolecules[molNum].GSmat 
        # myMolecules.theMolecules[molNum].GSenergy

@author: L. Andrew Wray
"""
import numpy as np
import copy
from math import floor

# p basis: '1' meaning p_z, '2' meaning p_x, '3' meaning p_y
# not relevant now, but for d/f use ml=0,1+-,2+-,3+- superposition order
atomNum=3

orbBasisDict={'s': ['s1'],
              'p': ['p1', 'p2', 'p3'],
              'd': ['d1', 'd2', 'd3', 'd4', 'd5'],
              'f': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']}



#####Define hopping matrices#####
#spin degree of freedom is excluded, but can be added with kron (see example below)

orbDict={'s': np.asarray([1]),
         'p_sig': np.asarray([1, 0, 0]),   # symmetry terms get 4 chars
         'p_pi1': np.asarray([0, 1, 0]),
         'p_pi2': np.asarray([0, 0, 1])}

#np.outer(orbDict['s'],orbDict['p']



#just give the off-diagonal pocket
orbHopDict={'ss_sig': np.zeros((1, 1)),  # symmetry terms get 4 chars
            'sp_sig': np.zeros((1, 3)),
            'pp_sig': np.zeros((3, 3)),
            'pp__pi': np.zeros((3, 3))}

hopType='ss_sig'
orbHopDict[hopType][0, 0] = 1
# orbHopDict[hopType]+=np.transpose(orbHopDict[hopType])
# orbHopDict[hopType]=np.kron(orbHopDict[hopType],np.eye(2)) #to include spin

hopType='sp_sig'
orbHopDict[hopType][0,0] = 1
# orbHopDict[hopType]+=np.transpose(orbHopDict[hopType])

hopType='pp_sig'
orbHopDict[hopType][0, 0] = 1
# orbHopDict[hopType]+=np.transpose(orbHopDict[hopType])

hopType='pp__pi'
orbHopDict[hopType][1:3,1:3] = np.eye(2)
# orbHopDict[hopType]+=np.transpose(orbHopDict[hopType])

orbDict={'s': np.asarray([1]),
         'p_sig': np.asarray([1,0,0]),
         'p_pi1': np.asarray([0,1,0]),
         'p_pi2': np.asarray([0,0,1])}

class SigmoidList:
    """Class storing a radially-resolved Hamiltonian parameter, framed as the sum of 
    2-parabola sigmoid-like functions.  The functions are defined as:
      -- for the first 1/2 of self._sWidth: 0.5*self.sAmps[pl]*((self._sStart[pl]-dVal)/(self._sWidth/2))**2
      -- for the next 1/2: self.sAmps[pl]*(1 - 0.5*((self._sStart[pl]-self._sWidth-dVal)/(self._sWidth/2))**2)
      -- the curve is flat outside this range
    Sigmoid functions have a 50% overlap with each of their nearest neighbors, 
    so that setting self.sAmps[:]=const gives a constant slope for the 
    Hamiltonian parameter
    
    Optimization operations on this list should include:
    1. Adding Val to self.Samp[pl] while subtracting Val from self.Samp[pl+1] (if pl+1 index is valid)
        -this allows for fine tuning of the large d parameters, while keeping small-d values fixed
    2. Same as (1), but subtract Val/N from pl+1...pl+N
    
    :ivar sNum: Number of sigmoids defining a curve
    :ivar dMin: Minimum radial distance - the curve is flat beyond this point
    :ivar dMax: Maximum radial distance - the curve is 0 beyond this point
    
    """
    
    def __init__(self,numSigmoids,dMin=0.5,dMax=6):
        self.sNum=numSigmoids
        self.dMin=dMin
        self.dMax=dMax
        
        #width of each sigmoid; place the near-field start in the middle of the last sigmoid: 
        self._sWidth=2*(dMax-dMin)/numSigmoids
       
        # a new sigmoid starts after each half-width:
        self._sStart=dMax-np.arange(numSigmoids)*(self._sWidth/2)

        #sigmoid centroids would be at self._sStart-self._sWidth/2
        
    def initAmpsLinear(self,nearFieldVal):
        #init sigmoid amplitudes:
        self.sAmps = np.ones(self.sNum)*nearFieldVal/(self.sNum-0.5)
        
        #define the alternative representations in terms of curve amplitude at the sigmoid centroids
        self.make_I_of_r()
    
    def initRanges(self,allMin=-0.5,allMax=2,absRange=0):
        """Define the allowed range of each parameter, for Monte Carlo sampling
        """
        
        if absRange: 
            self.max_sAmps=allMax * np.ones(self.sNum)
            self.min_sAmps=allMin * np.ones(self.sNum)
        else:
            self.max_sAmps=allMax*(self.readVal(0)/self.sNum) * np.ones(self.sNum)
            self.min_sAmps=allMin*(self.readVal(0)/self.sNum) * np.ones(self.sNum)
        
    def make_I_of_r(self): ####*****test
        """Create the curve contour self.I_of_r from sigmoid amplitudes self.sAmps.
        """
        self.I_of_r=np.zeros(self.sNum) # This function is doing double duty, both
                                        # initing the I_of_r variable and assigning
                                        # its values.  
        
        self.I_of_r[0]=self.sAmps[0]/2
        for pl in range(1,self.sNum):
            # each self.I_of_r increment is self.sAmps[pl-1]/2 + self.sAmps[pl]/2
            self.I_of_r[pl] = self.I_of_r[pl-1]+self.sAmps[pl-1]/2 + self.sAmps[pl]/2
        
    def make_sAmps(self):  ####*****test
        """Create sigmoid amplitudes (self.sAmps), given curve intensities at the sigmoid centroids

        :iVar I_of_r:  
        """
        
        self.sAmps[0]=2*self.I_of_r[0]
        for pl in range(1,self.sNum):
            # each self.I_of_r increment is self.sAmps[pl-1]/2 + self.sAmps[pl]/2
            self.sAmps[pl] = 2*(self.I_of_r[pl]-self.I_of_r[pl-1]-self.sAmps[pl-1]/2)
    
    def readVal(self,dist):
        """Output the curve amplitude for a given bond distance. Note that readVal
        references self.sAmps, so self.make_sAmps() must be called first if self.I_of_r
        is the vector being manipulated.
        
        :param dist: Distance from an atom.
        """
        
        # first set out-of-bounds values to the outermost in-bounds values
        if dist<=self.dMin:
            maxVal=np.sum(self.sAmps[:-1])+0.5*self.sAmps[-1]
            return maxVal
        elif dist>=self.dMax:
            return 0
        
        #now deal with in-bounds cases        
        else:
            # first frame the distance from the far-field, in units of _sWidth:
            normDist=(self.dMax-dist)/self._sWidth
            
            # value contribution from full sigmoids:
            fullSigmoids=floor(2*normDist-1)
            fullSigmoids=fullSigmoids*(fullSigmoids>0)
            # disp(fullSigmoids)
            # self.fullSigmoids=fullSigmoids
            theVal=np.sum(self.sAmps[:fullSigmoids])
            
            #now add the partial sigmoid component:
            if normDist>0.5:   # if we're considering the overlap of 2 sigmoids:  
                #define partial dist in units of self._sWidth/2
                partialDist=2*(normDist-(fullSigmoids+1)/2)
                # print(partialDist)
                #up parabola
                theVal+=0.5*self.sAmps[fullSigmoids+1] * partialDist**2
                
                #down parabola
                theVal+=self.sAmps[fullSigmoids] * (1 - 0.5*(1-partialDist)**2)
            else:
                partialDist=2*normDist
                # print(partialDist)
                #up parabola only
                theVal+=0.5*self.sAmps[fullSigmoids] * partialDist**2
            
            return theVal




#create a list of interactions in the sigmoid model

umHopTerms=[]
umIntTerms=[] #not yet implemented
umIntTerms_sameAtom=[] #not yet implemented
umAtomAtomIntTerms=[]
umAtomOrbIntTerms=[]  #one atom to orbitals on another atom (in hopping hamiltonian)
umOrbitalEnergies=[]
maxDist=8 #Angstroms; maximum allowed distance for interacting atoms

#other terms?  Have hopping change with density?
sigmoidNum=5

####first init hopping:
#***probably need to specify orbTypes too -- look at state2num
theTerm=SigmoidList(sigmoidNum)
theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
theTerm.orbSyms=['s','s']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(-3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['C','C']
theTerm.orbSyms=['s','p']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['C','C']
theTerm.orbSyms=['p','p']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['C','C']
theTerm.orbSyms=['p','p']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'__pi'
theTerm.initAmpsLinear(-1.5)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']
theTerm.orbSyms=['s','s']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(-3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']
theTerm.orbSyms=['s','p']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','H']
theTerm.orbSyms=['s','s']
theTerm.symName=theTerm.orbSyms[0]+theTerm.orbSyms[1]+'_sig'
theTerm.initAmpsLinear(-3)
theTerm.initRanges()
umHopTerms+=[copy.deepcopy(theTerm)]

###next define orbital energies (single atom)
#***make a separate list for orbital energies - these may not need to be trained,
#as that can be handled by the E(n,) curve


#***for interactions, init sigmoid E(n,atom1,orb1,sym1,atom2,orb2,sym2)
   #***atom/orb/sym_1==..._2 is OK, but need to mitigate self-interaction in the implementation
   #*** also, for atom/orb/sym_1==..._2, the sigmoid argument is density (i.e.)

#***possibilities for dealing with atom-atom interactions:
#1. static Coulomb term outside of the electronic sector (probably a good idea!)
#2. static Coulomb term acting on orbital energies, but with no charge density component



theTerm=['H','s',0,[-1,1]]
umOrbitalEnergies+=[theTerm]
theTerm=['C','s',-3,[-4,-2]]
umOrbitalEnergies+=[theTerm]
theTerm=['C','p',-1,[-2,0]]
umOrbitalEnergies+=[theTerm]

###next define Atom-Atom interactions

theTerm=SigmoidList(sigmoidNum)
theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
theTerm.initAmpsLinear(0)
theTerm.initRanges(-5/sigmoidNum,5/sigmoidNum,1)
umAtomAtomIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']  #CC s-s orbital hopping
theTerm.initAmpsLinear(0)
theTerm.initRanges(-5/sigmoidNum,5/sigmoidNum,1)
umAtomAtomIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','H']  #CC s-s orbital hopping
theTerm.initAmpsLinear(0)
theTerm.initRanges(-5/sigmoidNum,5/sigmoidNum,1)
umAtomAtomIntTerms+=[copy.deepcopy(theTerm)]

###next define [Atom1]-->[Atom2 orbital] interactions
#These should include strong short-range repulsion if there are no density-density interactions

theTerm=SigmoidList(sigmoidNum)
theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
theTerm.orbSyms='s'
theTerm.symName='_sig'
theTerm.initAmpsLinear(2)
theTerm.initRanges()  #Need to adjust this
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
theTerm.orbSyms='p'
theTerm.symName='_sig'
theTerm.initAmpsLinear(2)
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
theTerm.orbSyms='sp'
theTerm.symName='_sig'
theTerm.initAmpsLinear(1) #may want to change so that sp amplitude is a multiplier related to the 's' and 'p' amplitudes 
# theTerm.referenceTerms=[0,1]  # define amplitude in terms of umAtomOrbIntTerms[theTerm.referenceTerms[0]] and umAtomOrbIntTerms[theTerm.referenceTerms[1]]
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm=SigmoidList(sigmoidNum)
theTerm.atomTypes=['C','H']  #CC s-s orbital hopping
theTerm.orbSyms='s'
theTerm.symName='_sig'
theTerm.initAmpsLinear(2)
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']  #CC s-s orbital hopping
theTerm.orbSyms='s'
theTerm.symName='_sig'
theTerm.initAmpsLinear(2)
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']  #CC s-s orbital hopping
theTerm.orbSyms='p'
theTerm.symName='_sig'
theTerm.initAmpsLinear(2)
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]

theTerm.atomTypes=['H','C']  #CC s-s orbital hopping
theTerm.orbSyms='sp'
theTerm.symName='_sig'
theTerm.initAmpsLinear(1) #may want to change so that sp amplitude is a multiplier related to the 's' and 'p' amplitudes 
# theTerm.referenceTerms=[0,1]  # define amplitude in terms of umAtomOrbIntTerms[theTerm.referenceTerms[0]] and umAtomOrbIntTerms[theTerm.referenceTerms[1]]
theTerm.initRanges()
umAtomOrbIntTerms+=[copy.deepcopy(theTerm)]




###next init density-density interactions 
# theTerm=SigmoidList(sigmoidNum)
# theTerm.atomTypes=['C','C']  #CC s-s orbital hopping
# theTerm.mat=orbHopDict['ss_sig']   #not done yet***
# theTerm.initAmpsLinear(-3)
# theTerm.initRanges()
# HmatTerms+=[theTerm]


#***add same-atom density-density interactions

#***add the sp sigma_x-like term


##### functions to create the tight binding Hamiltonian for a given universal model#####

# theMol.findPairs(maxDist)                       # 1
# createHoppingHmat(theMol,umHopTerms,umOrbitalEnergies)   # 2
# addOrbitalEnergies(theMol,umOrbitalEnergies)   # 3
# addAtom_OrbInt(theMol,umAtomOrbIntTerms)       # 4

    
def createHoppingHmat(theMol,umHopTerms,umOrbitalEnergies):
    #first clear the hopping Hamiltonian
    #need to call theMol.findPairs(maxDist) first
    theMol.theHopHmat=np.zeros((theMol.totalBasisSize,theMol.totalBasisSize))
    
    #now loop through the pairs list and add the terms
    for ii in range(len(theMol.pairs_list)):
        addPairwiseHopping(theMol,umHopTerms,theMol.pairs_list[ii][0],theMol.pairs_list[ii][1])
    
    addOrbitalEnergies(theMol,umOrbitalEnergies)

def addOrbitalEnergies(theMol,umOrbitalEnergies):
    #add orbital energies (diagonal terms) from umOrbitalEnergies to theMol.theHopHmat
    
    for orbType in umOrbitalEnergies:
        #first find all corresponding atoms
        theInds=np.where(np.asarray(theMol.atom_names)==orbType[0])[0]
        for atomNum in theInds:
            basisInd=theMol.atomBasisLocs[atomNum][orbType[1]]
            
            theLim=len(orbBasisDict[orbType[1]])
            theMol.theHopHmat[basisInd:basisInd+theLim,basisInd:basisInd+theLim] += orbType[2]*np.eye(theLim)
            
def addAtom_AtomInt(theMol,umAtomAtomIntTerms): 
    # Add pairwise atomic interactions based only on atom type (not charge density-resolving).
    # The result is a single energy value added to all diagonal Hamiltonian matrix elements
    
    theEnergy=0
    for listPair in theMol.pairs_list:
        theAtoms=[theMol.atom_names[listPair[0]],theMol.atom_names[listPair[1]]]
        for intTerm in umAtomAtomIntTerms:
            if theAtoms==intTerm.atomTypes:
                theEnergy+= intTerm.readVal(listPair[2])
                break
    
    theMol.theHopHmat+=theEnergy*np.eye(theMol.totalBasisSize)
    
def addAtom_OrbInt(theMol,umAtomOrbIntTerms): 
    # Add pairwise atom-orbital interactions based only on atom type (not charge density-resolving).
    # Note that this term does not check the atom type -- be careful to call it with a valid term.
    
    for listPair in theMol.pairs_list:
        theAtoms=[theMol.atom_names[listPair[0]],theMol.atom_names[listPair[1]]]
        for intTerm in umAtomAtomIntTerms:
            if theAtoms==intTerm.atomTypes:
                addOrbPert(theMol,intTerm,listPair)
            if theAtoms[::-1] ==intTerm.atomTypes:
                addOrbPert(theMol,intTerm,listPair[::-1])
                
def addOrbPert(theMol,intTerm,listPair):
    
    #first find the start of the basisPl
    
    basisStart=theMol.atomBasisLocs[listPair[1]][intTerm.orbSyms[0]]
    pertVal=intTerm.readVal(listPair[2])
    
    #now insert sigma terms or sp term
    if intTerm.orbSyms=='s':
        theMol.theHopHmat[basisStart,basisStart]+=pertVal
    else:                 #generate rotation matrix for p-orbitals
        bondVector=theMol.coords_arry[listPair[0]]-theMol.coords_arry[listPair[1]] #pointing towards the perturbation

        if intTerm.orbSyms=='p':
            rotMat=rotateBasis(intTerm.orbSyms[0], bondVector)[0]
            
            if intTerm.symName[-4:] == '_sig': #The only scenario considered 
                theMol.theHopHmat[basisStart:basisStart+3,basisStart:basisStart+3] += pertVal*np.matmul(rotMat.T,np.matmul(orbHopDict['pp_sig'],rotMat))
            else:
                print('Error 001, non-implemented interaction.')
        elif intTerm.orbSyms=='sp':
            rotMat=rotateBasis(intTerm.orbSyms[1], bondVector)[0]
            if intTerm.symName[-4:] == '_sig': #The only scenario considered 
                theMol.theHopHmat[basisStart,basisStart+1:basisStart+4] +=  pertVal*np.matmul(orbDict['p_sig'],rotMat)
                theMol.theHopHmat[basisStart+1:basisStart+4,basisStart] +=  pertVal*np.matmul(rotMat.T,orbDict['p_sig'])
            else:
                print('Error 001, non-implemented interaction.')

def addPairwiseHopping(theMol,umHopTerms,atom1num,atom2num,forceAmp=[]):
    # use forceAmp to disregard the mm amplitude and use indicated value
    
    bondVector=theMol.coords_arry[atom2num]-theMol.coords_arry[atom1num]
    if np.linalg.norm(bondVector) > maxDist:
        return
    else:
            
        element1,element2 = theMol.atom_names[atom1num] , theMol.atom_names[atom2num]
        
        for hopTerm in umHopTerms:
            if set(hopTerm.atomTypes)=={element1,element2}:
                if (element2==hopTerm.atomTypes[0]) and (element1 != hopTerm.atomTypes[0]):
                    # swap atom1 and atom2 if the atoms were reversed relative to the hopterm
                    tmp=atom2num
                    atom2num=atom1num
                    atom1num=tmp
                    tmp=element2
                    element2=element1
                    element1=tmp     
                    bondVector=-bondVector
                    
                # find orbital types, confirm the elements have them, and add the hopping matrix
                orbType1,orbType2=hopTerm.orbSyms[0],hopTerm.orbSyms[1]
                
                if element1==element2 and orbType1!=orbType2:
                    configNum=2 #there are 2 equivalent configurations (e.g. C1_s<-->C2_p and C1_p<-->C2_s)
                else: configNum=1
                
                while configNum>0:
                    basisInd1,basisInd2= theMol.atomBasisLocs[atom1num][orbType1],theMol.atomBasisLocs[atom2num][orbType2] #basis place
                    if basisInd1 >=0 and basisInd2>=0:  #if neither orbital has been excluded, put in the hopping terms
                        hopMat1Term=copy.deepcopy(orbHopDict[hopTerm.symName])
                        #now rotate hopMat1Term:
                        rotMat2,xHat = rotateBasis(orbType2, bondVector)
                        hopMat1Term=np.matmul(hopMat1Term,rotMat2)
                        hopMat1Term=np.matmul(np.transpose(rotateBasis(orbType1, bondVector,xHat)[0]),hopMat1Term)
                        
                        #now multiply by the needed amplitude
                        if forceAmp==[]:
                            hopMat1Term*=hopTerm.readVal(np.linalg.norm(bondVector))
                        else:
                            hopMat1Term*=forceAmp
                            
                        #now enter the hopping term in the matrix:
                        theLims=orbHopDict[hopTerm.symName].shape
                        theMol.theHopHmat[basisInd1:basisInd1+theLims[0],basisInd2:basisInd2+theLims[1]] += hopMat1Term
                        #now add the conjugate transpose
                        theMol.theHopHmat[basisInd2:basisInd2+theLims[1],basisInd1:basisInd1+theLims[0]] += np.conj(np.transpose(hopMat1Term))
                    
                    configNum-=1
                    if configNum==1: #swap atoms for the reverse orbital configuration 
                        tmp=atom2num
                        atom2num=atom1num
                        atom1num=tmp
                        bondVector=-bondVector
                
                #add terms directly to theHopHmat
        #find the relevant hopping terms
        
        #loop through the orbital types
            
        #rotate the basis
        
        #insert the hopping terms with appropriate weight
    
#def createTermHopMat(theMol,theHopTerm)
   #createTermHopMat(theMol,umHopTerms[1])

#####Now define basis transformation upon rotation#####
def rotateBasis(orbType, bondVector, new_xhat=[]):
    # currently only defined for s- and -p orbitals
    # returns rotMat such that orbHmat'=np.matmul(np.transpose(rotMat),np.matmul(orbHmat,rotMat))
    
    #returns a matrix that changes the orbital basis from zhat=[001] to zhat=bondVector
    # the new xhat can be specified, or will 
    # orbType=='s' will return a 2x2 identity matrix
    
    assert orbType=='s' or orbType=='p', "Expected s- or p- orbital"
    
    if orbType=='p':
        new_zhat=copy.copy(bondVector)
        new_zhat/=np.linalg.norm(new_zhat)
        
        if new_xhat==[]: #seed with an arbitrary orientation 
            new_xhat=np.random.rand(3)
        # make sure xhat is normal to pz:
        new_xhat=np.cross(new_xhat,new_zhat)
        new_xhat=np.cross(new_zhat,new_xhat)
        
        new_xhat/=np.linalg.norm(new_xhat)
        new_yhat=-np.cross(new_xhat,new_zhat)   
        
        #transition from vector space to orbital basis space:
        new_xhat_orb=np.asarray([new_xhat[2],new_xhat[0],new_xhat[1]])
        new_yhat_orb=np.asarray([new_yhat[2],new_yhat[0],new_yhat[1]])        
        new_zhat_orb=np.asarray([new_zhat[2],new_zhat[0],new_zhat[1]])   

        #create the change of basis matrices        
        pxMap=np.outer(orbDict['p_pi1'],new_xhat_orb)
        pyMap=np.outer(orbDict['p_pi2'],new_yhat_orb)
        pzMap=np.outer(orbDict['p_sig'],new_zhat_orb)
        #***modify with kron(...,np.eye(2)) if converting to a spinful represention (not implemented)
        rotMat=pxMap+pyMap+pzMap
        
    elif orbType=='s':
        rotMat=np.eye(1)
        
    return rotMat, new_xhat
    


    
    
    
    
    

def listOrbBasis(atomNum,orbLetter):
    #list out the sub-basis of a single-atom orbital manifold
    num2state=[]
    for pl in range(len(orbBasisDict[orbLetter])):
        num2state+=[[atomNum, orbBasisDict[orbLetter][pl]]]

    return num2state





# class HamBasis:
#     def __init__(self):
        
#         self.ppSigma=




#to do:
    #1. debug oneMol, add num2state list
    #2. define Hamiltonian matrices and create a framework that will
    # play well with added elements in the future
    
    
