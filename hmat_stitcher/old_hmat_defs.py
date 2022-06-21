#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:04:28 2022

@author: lawray
"""

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