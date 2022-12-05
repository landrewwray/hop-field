#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:22:57 2022

@author: lawray

TO DO:
    
-Appears to be a significant overfitting issue for low temperature Monte Carlo convergence???
-economize memory usage.  For example, oldValList terms can be empty when not in use?


"""

import copy
import numpy as np
import monte_carlo.make_hmat as mkh
import monte_carlo.summarize_mc as smc

#from numba import njit, int32, float64
#from numba.experimental import jitclass

maxDisplacement = 50 # Units of the displacement size


def randUpdate(termCurves,termNums,isSingleAtom):
    # just return the updated termCurve and termNum
    
    # choose a number to update    
    selectedTerm = np.random.randint(len(termNums))
    selectedNum = np.random.randint(termNums[selectedTerm].shape[0])
    
    #choose a new value for it
    valAmp = termNums[selectedTerm][selectedNum][1]-termNums[selectedTerm][selectedNum][2]
    newVal = valAmp*np.random.rand() + termNums[selectedTerm][selectedNum][2]
    
    #now update the terms:
    newTermNum = copy.deepcopy(termNums[selectedTerm])
    newTermNum[selectedNum,0] = newVal
        
    if not isSingleAtom[selectedTerm]:    
        newTermCurve = termCurves[selectedTerm].copy()
        newTermCurve.I_of_r[selectedNum] = newVal
        newTermCurve.make_sAmps()
    else:
        newTermCurve = newTermNum[0].copy()
        
    return newTermCurve, newTermNum, selectedTerm
    
def posToEnergy(posVect, testLists=[], optimizeAll=False):
    """Convert a position vector to a Monte Carlo 'energy' for evaluating steps.
    
    If optimizeAll=False, then only posList coordinates listed in testLists[0]
    will contribute to the Monte Carlo evaluation.  Otherwise,  all of the 
    testLists will be summed over for the MC energy (newEnergy).
    """
    
    if len(testLists) == 0:  # dummy init:
        testLists = [np.asarray(range(len(posVect)), dtype=np.int32)]
    
    posTmp= np.abs(posVect)
    
    posTmp[np.where(posTmp > maxDisplacement)[0]] = maxDisplacement
    
    listNum=0
    trackedEnergies = np.zeros(len(testLists))
    angle_length_error = np.zeros((2,len(testLists)))
    
    for trackedList in testLists:
        trackedEnergies += [0]
        
        numLen=0
        numAng=0
        normVal=1/(maxDisplacement)
        for posPl in trackedList:
            trackedEnergies[listNum] += posTmp[posPl]
            if posPl%2 == 0:  #angle error
                angle_length_error[0, listNum] += posTmp[posPl]*0.02*180/np.pi  # error in degrees, approximating
                numAng += 1                                                  # a bond length of 1
            else:
                angle_length_error[1, listNum] += posTmp[posPl]*0.01  # error in Angstroms
                numLen += 1

        # now normalize and increment listNum:
        trackedEnergies[listNum] *= normVal        
        if numAng>0:
            angle_length_error[0,listNum] /= numAng
        if numLen>0:
            angle_length_error[1,listNum] /= numLen
        
        listNum+=1

    if optimizeAll:
        newEnergy = sum(trackedEnergies)
    else:
        newEnergy = trackedEnergies[0]
    
    return newEnergy, trackedEnergies, angle_length_error


def getEnergy(hmat, numElectrons):
    """Obtain the energy of a molecule from hmat and the electron filling. The
    current version uses full diagonalization, as >~50% of states are occupied.
    
    TO DO:
    -explore Lanczos and/or lower precision for a speed boost.  Note that there are
    fewer unoccupied states, and it is equivalent to solve for those if only the 
    HF ground state energy is needed.
    """
    
    stateEnergies=np.real(np.linalg.eig(hmat)[0])
    stateEnergies.sort()
    
    numFilledStates = int(np.floor(numElectrons/2))
    totalEnergy = sum(stateEnergies[:numFilledStates]) * 2

    if (numElectrons % 2): # if there's a half-filled level
        totalEnergy += stateEnergies[numFilledStates]

    return totalEnergy

def getEvals(hmatList,numElectrons):
    """get the molecular energy for each hmat.
    
    TO DO:
    -This operation is expensive.  For a future update, it may be useful to 
    have the option to just sample a subset of the hmats.    
    """
    energiesList=[]
    
    for molPl in range(len(hmatList)):
        energiesList += [[]]
        
        for bondPl in range(len(hmatList[molPl])):
            energiesList[molPl] += [[]]
            
            for distPl in range(len(hmatList[molPl][bondPl])):
                energiesList[molPl][bondPl] += [[]]
                
                if distPl > 0 or bondPl == 0:
                    energiesList[molPl][bondPl][distPl] = getEnergy(hmatList[molPl][bondPl][distPl],numElectrons[molPl])
                    
                else:
                    energiesList[molPl][bondPl][0] = energiesList[molPl][0][0]
    
    return energiesList

def energies2posVect(energiesList):
    """Return the 
    
    posVect is in units of the unidirectional distortion size
    """
    

    posVect = []
    
    for molPl in range(len(energiesList)):
        for bondPl in range(len(energiesList[molPl])):
            for distPl in range(1,len(energiesList[molPl][bondPl]),2):
                #select the first of each distortion pair
                
                #now assign the distance
                slope1 = energiesList[molPl][bondPl][distPl+1] - energiesList[molPl][bondPl][0]
                slope0 = energiesList[molPl][bondPl][0] - energiesList[molPl][bondPl][distPl]
                
                if slope1-slope0 > 0:
                    posVect += [np.mean([slope0,slope1]) / (slope1-slope0)]
                    
                    # overwrite extremely large values:
                    if abs(posVect[-1]) > maxDisplacement:
                        posVect[-1] = maxDisplacement * posVect[-1]/abs(posVect[-1])
                else: # if the potential is parabolic-down:
                    posVect += [maxDisplacement]
                
    return posVect

def getPosVect(hmatList,numElectrons):
    """Return a list of displacement from ideal coordinates for the selected bond
    distortions. (this is the 'y' for our discriminator model)
    
    """
    
    energiesList = getEvals(hmatList,numElectrons)
    
    posVect = energies2posVect(energiesList)
    
    return posVect


def updateUMterms(termNums,termCurves,newTermCurve,newTermNum,selectedTerm):
    """Swap in the new UM terms, and save the old ones.
    """
    
    oldTermNum = copy.deepcopy(termNums[selectedTerm])
    oldTermCurve = termCurves[selectedTerm].copy()

    termCurves[selectedTerm] = newTermCurve.copy()
    termNums[selectedTerm] = copy.deepcopy(newTermNum)

    return oldTermCurve, oldTermNum 


def proposeStep(hmatList, mcLists, valList, oldValList, termCurves, termNums):
    """Propose a new UM, with one modified term.  The return includes everything needed
    to evaluate the step, including the new term values, the new Hamiltonians+energies,
    and the new posVect.
    
    If the new UM is accepted, one can simply save and continue; otherwise, one needs to
    undo the step with a separate function.
    

    Parameters
    ----------
    mcLists : TYPE
        DESCRIPTION.
    valList : TYPE
        DESCRIPTION.
    oldValList : TYPE
        DESCRIPTION.

    Returns
    -------
    termPl : int
        The number of the modified term
    selectedTerm : int
        Number of the modified term
    oldTermCurve
        Needed for undoStep
    oldTermNum
        Needed for undoStep
    
    proposedValList
    Return the new valList,

    """    
    # 1. generate the term update
    newTermCurve, newTermNum, selectedTerm = randUpdate(termCurves,termNums,mcLists.isSingleAtom) # time this later!
    
    # 2. update valList and oldValList; use them to update the Hmat

    # first the term lists:
    oldTermCurve, oldTermNum = updateUMterms(termNums,termCurves,newTermCurve,newTermNum,selectedTerm)
    
    # then the valList
    oldValList[selectedTerm] = copy.deepcopy(valList[selectedTerm])    
    valList[selectedTerm] = mkh.makeHmatValList(selectedTerm, mcLists, termCurves)
           
    # now the Hmat
    mkh.addTermHmatMEs(selectedTerm, hmatList,valList,oldValList,mcLists)
    
    return selectedTerm, oldTermCurve, oldTermNum


def undoStep(hmatList, mcLists, valList, termNums, termCurves, oldValList, oldTermCurve, oldTermNum, selectedTerm):
    """Reverts the lists as follows:
    1. Undo term curve/num changes
    2. Swap in the oldValList term to newValList
    3. update the hamiltonians
    """
    #1. Undo term curve/num changes
    termCurves[selectedTerm] = oldTermCurve.copy()
    termNums[selectedTerm] = copy.deepcopy(oldTermNum)
    
    #2. Swap in the oldValList term to newValList
    triedVals = valList[selectedTerm].copy()
    valList[selectedTerm] = oldValList[selectedTerm].copy()
    oldValList[selectedTerm] = triedVals
    
    # 3. update the hamiltonians
    mkh.addTermHmatMEs(selectedTerm, hmatList, valList, oldValList, mcLists)

def generateTval(hmatList, valList, oldValList, termCurves, termNums, mcLists,samplesPerTrial, numSteps = 20):
    """ Not currently in use.  Please disregard this function.
    """
    # first obtain the position vector for hmatList (list of float64)
    # infinities should be avoided...
    
    posVect = getPosVect(hmatList,mcLists.numElectrons)
    print('posVect')
    print(posVect)
    
    # translate position vector into an 'energy'. This will define the maximum position
    energyStart = posToEnergy(posVect)
    print('energyStart')
    print(energyStart)
    
    energyList = [energyStart]
    for pl in range(numSteps):
        selectedTerm, oldTermCurve, oldTermNum = proposeStep(hmatList, mcLists, valList, oldValList, termCurves, termNums)
        
        posVect = getPosVect(hmatList,mcLists.numElectrons)
        print('newPosVect')
        print(posVect)
        
        newEnergy = posToEnergy(posVect)
        
        print('newEnergy (' + str(pl) + '):')
        print(newEnergy)
        
        energyList += [newEnergy]

        if abs(energyList[-1]-energyList[0])>1e-7:
            print('here!!!!!')
            print('here!!!!!')
            print('here!!!!!')
            
        undoStep(hmatList, mcLists, valList, termNums, termCurves, oldValList, oldTermCurve, oldTermNum, selectedTerm)
    
    #define the starting temperature to allow a certain fraction of steps, avoiding large-valued outliers
    acceptFrac = 0.5
    energyList.sort()
    Tval = energyList[round(acceptFrac*numSteps)]-energyList[0]

    # Tval = 2*np.std(energyList)
    
    # now update hmatList
    

    # now evaluate the     
    
    
    # add a constant, to deal with the possibility of Tval~0

    return Tval

def makeMolTestLists(test1mc0, mcLists):
    """ Returns lists of posVect indices, as defined by test1mc0. The test1mc0
    list uses values of 0 for molecules involved in Monte Carlo optimization, and
    >=1 for molecules that should be tracked but not optimized on.  For example,
    this defines a Monte Carlo optimization of the first 3 molecules, which will
    separately track quality factors for molecules # 4-5 and molecule #6-8:

    test1mc0=np.asarray([0,0,0,1,1,2,2,2])
    """
    
    test1mc0.astype(np.int32)
    
    #first map the test1mc0d values onto the posVect entries:
    pos_test_state = []
    dummyHmatList = mkh.generate_empty_HmatList(mcLists)
    for molPl in range(len(dummyHmatList)):
        for bondPl in range(len(dummyHmatList[molPl])):          
            pos_test_state += [test1mc0[molPl],test1mc0[molPl]]
    pos_test_state = np.asarray(pos_test_state)
    print(pos_test_state)
    #now create lists of the relevant indices for each 
                
    testInds=list(set(test1mc0))
    testInds.sort()
    
    testLists=[]
    for testInd in testInds:
        testLists += [np.where(pos_test_state == int(testInd))[0]]

    return testLists

def runMonteCarlo(convergenceCycles, Tval, temperatureStages, decayPerT, stepsPerT, hmatList, valList, oldValList, termCurves, termNums, mcLists, testLists = [], optimizeAll=False):
    """Runs a Monte Carlo simulated annealing exploration of the universal model parameter space.
    
    Note that floating point error in each element of posVect accumulates at a rate of ~10**-12 in 1500 
    attempted steps with 20 terms. This can be fixed by re-initializing the Hamiltonian with
    mkh.makeAllHmats(mcLists, termCurves), but it is very unlikely to be relevant. (should grow as 
    sqrt(steps))
    
    Use testLists and optimizeAll=False to separate training and test scenarios.
    """

    
    
    posVect = getPosVect(hmatList,mcLists.numElectrons)
    print('posVect')
    print(posVect)
    
    # translate position vector into an 'energy'. This will define the maximum position
    energyStart, trackedEnergies, angle_length_error = posToEnergy(posVect, testLists,optimizeAll)
    print('energyStart')
    print(energyStart)

    # track the evolution:
    termHist = [copy.deepcopy(termNums)]
    vectHist = [copy.deepcopy(posVect)]
    energyHist = [copy.copy(energyStart)]
    trackedEnergyHist=[trackedEnergies]
    angle_length_errorHist = [angle_length_error]         

    energyList = [energyStart]
    goodMoves = [1] # count the initial state as a 'good move'
    
    for cyclePl in range(convergenceCycles):
        TvalNow = Tval
        for tempPl in range(temperatureStages):
            print('Current temperature is: ' + str(TvalNow))
        
            for pl in range(stepsPerT):
                selectedTerm, oldTermCurve, oldTermNum = proposeStep(hmatList, mcLists, valList, oldValList, termCurves, termNums)
                
                posVect = getPosVect(hmatList,mcLists.numElectrons)
                # print('newPosVect')
                # print(posVect)
                
                newEnergy, trackedEnergies, angle_length_error = posToEnergy(posVect,testLists,optimizeAll)
                
                # print('newEnergy (' + str(pl) + '):')
                # print(newEnergy)
                
                # track the evolution:
                termHist += [copy.deepcopy(termNums)]
                vectHist += [copy.deepcopy(posVect)]
                energyHist += [copy.copy(newEnergy)]
                trackedEnergyHist += [trackedEnergies]
                angle_length_errorHist += [angle_length_error]      
                
                expFact=-(newEnergy - energyList[-1])/TvalNow
                if abs(expFact)>20:  #avoid exp overflow error
                    expFact = 20*expFact/abs(expFact)                
                qualFact = np.exp(expFact)
                
                if not (np.random.rand() < qualFact):
                    undoStep(hmatList, mcLists, valList, termNums, termCurves, oldValList, oldTermCurve, oldTermNum, selectedTerm)
                    goodMoves += [0]
                else:
                    goodMoves += [1]
                    energyList += [newEnergy]
                    currentPosVect = posVect
                    
            TvalNow *= decayPerT
        
    print('\n\nFinal posVect:')
    print(currentPosVect)
    
    mc_path = [termHist,vectHist,energyHist]
    trackedEnergyHist=np.asarray(trackedEnergyHist)
    angle_length_errorHist=np.asarray(angle_length_errorHist)
    
    mc_data = smc.MonteCarloOutput(energyList, goodMoves, mc_path, trackedEnergyHist, angle_length_errorHist)
    
    return mc_data

