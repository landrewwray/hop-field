#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 22:10:56 2022

@author: lawray

Monte Carlo-based data generator for the model defined in model_creation

TO DO:
-allow selection of sigmoid amplitude (sAmp) or curve amplitude (I_of_r) as 
the Monte Carlo parameter
-consider forcing small incremental motion for parameters (e.g. within +-0.2 eV)
-imaging of single/double bond accuracy vs time in Monte Carlo

Testing/debugging:
-

Project planning:
-contact NYU speaker and others in molecular dynamics/ML - schedule quick meetings
-is longer range hopping connectivity only relevant if it changes the number of
single vs double bonds? (is there sometimes/often an accurate local picture?)
   -what is the energy scale of quantum well effects in relevant scenarios?

"""

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

import monte_carlo.monte_carlo as mc
import monte_carlo.decant_terms as dec
import monte_carlo.make_hmat as mkh

#import time
# import monte_carlo.make_hmat as mkh

################################################################
################### 2. Define MC parameters ####################
################################################################

saveName = 'Config Saves/trial2.p2'
with open(saveName, "rb") as f:
    mcLists = pickle.load(f)

termCurves = dec.decant_UM_terms(mcLists.theUM) # numba SigmoidCurve curves and single atom values
termNums = dec.readTerms(termCurves) # just the underlying parameters

################################################################
###################### 1. Load the model #######################
################################################################

slowConverge=False
if slowConverge:
    convergenceCycles = 5 # start over this many times
    temperatureStages = 5
    decayPerT = 0.3  # multiply this onto temperature with each step
    stepsPerT = 1000  # ideally want ~>10000
    numResets=10  # to run Monte Carlo multiple times and explore multiple convergence paths
else:
    convergenceCycles = 3
    temperatureStages=3
    decayPerT = 0.1  # multiply this onto temperature with each step
    stepsPerT = 100  # ideally want ~>10000
    numResets=0  # to run Monte Carlo mult

test1mc0=np.zeros(len(mcLists.numElectrons))
test1mc0[0:2] = 1 # exclude these from the Monte Carlo optimization
test1mc0[6:8] = 2 # exclude these from the Monte Carlo optimization
testLists = mc.makeMolTestLists(test1mc0, mcLists)

initialTmultiplier = 1
Tval = 2 # or define using:
    # Tval = mc.generateTval(hmatList, valList, oldValList,termCurves, termNums, mcLists,samplesPerTrial,numSteps=20)
    # Tval *= initialTmultiplier

################################################################
################### 3. Create initial Hmats ####################
################################################################

hmatList, valList, oldValList = mkh.makeAllHmats(mcLists, termCurves) #***** test!!!

# Tval = mc.generateTval(hmatList, valList, oldValList,termCurves, termNums, mcLists,samplesPerTrial,numSteps=20)
# Tval *= initialTmultiplier


run_MC = True
if run_MC:
    time0=time.time()
    mc_data = mc.runMonteCarlo(convergenceCycles, Tval, temperatureStages, decayPerT, stepsPerT, hmatList, valList, oldValList,termCurves, termNums, mcLists, testLists)
    time1=time.time()
    print('Monte Carlo runtime is ' + str(time1-time0) + ' s, or ' + str((time1-time0)/(stepsPerT*temperatureStages)) + 's per attempted step.')
    
    mc_data.makeTrainingData()
    mc_data.save()
    # plt.plot(np.log(np.asarray(energyList)))
    # plt.plot(np.asarray(energyList))
    
    # for pl in range(trackedEnergyHist.shape[1]):
    #     plt.plot(np.arange(trackedEnergyHist.shape[0]),trackedEnergyHist[:,pl])
    # plt.show()

print("Now test for numerical drift:")
hmatList, valList, oldValList = mkh.makeAllHmats(mcLists, termCurves) #***** test!!!
posVect = mc.getPosVect(hmatList,mcLists.numElectrons)
print('posVect')
print(posVect)
