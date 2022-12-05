#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022

This code defines the molecules, model parameters, and bond distortions, and uses
these to create hamiltonian symmetry factors.

"""

import pickle
import time

import universalmodel.universalmodel as um
import load_create_mol.load_mol as load_mol
import hmat_stitcher.hmat_stitcher as stitch

import monte_carlo.decant_terms as dec

################################################################
#### 1. code to load and create molecule --> molConfigList  ####
################################################################

# molFilePath="Molecular Structure Data/*.mol2"
molFilePath="MolSD CNO/*.mol2"
bondsPerMol=5 # Number of bonds to distort in each molecule (if enough bonds exist)
moleculesToLoad=[]  # use [] to load all molecules, and 0 entries
                        # to not load a particular molecule
sizeCap=35  # Only load molecules with <=sizeCap atoms. Set to sizeCap<=0 to disable
            # this filter

makeNewDistortions = True
if makeNewDistortions:
    structInfo = load_mol.load_struct_info(molFilePath,bondsPerMol,sizeCap, moleculesToLoad)
# configs contains: atomsLists, bondsArrays, coordsArrays, distortLists, molNumList, elementsLists

################################################################
################ 2. make the universal model ###################
################################################################

theUM = um.UniversalModel(['C','Cinit_0.txt'])
theUM += um.UniversalModel(['H','Hinit_0.txt'])  
theUM += um.UniversalModel(['O','Oinit_0.txt'])

theUM.makeCrossTerms(['all'],'Hop_generic0.txt')
theUM.popHs() # remove the hydrogen s-orbital to eliminate the scalar shift degree of freedom


# theUM.popHop()     # eliminate all hopping terms (classical 2-body model!)
# theUM.popHop('H')  # eliminate just hopping to/from Hydrogen atoms


################################################################
############### 3. create matrix element lists #################
################################################################

loadTheFile=False
if loadTheFile:
    actionWordStem='Load'
else:
    actionWordStem='Creat'
print(actionWordStem + "ing hamiltonian symmetry terms.")

start_time=time.time()

if not loadTheFile:
    saveName='Config Saves/trial2.p'
    
    symMats=stitch.SymTerms(theUM,structInfo) #working, but need to continue testing the output!
    
    # stitch.mergeLikeMats(symMats)  #*** Not currently functional - sometimes breaks the
                                     # equivalency of term length for distortions
    
    with open(saveName, "wb") as f:
        pickle.dump(symMats, f)
        
else:
    saveName='Config Saves/trial2.p'
    
    with open(saveName, "rb") as f:
        symMats = pickle.load(f)

finish_time=time.time()
print("Symmetry terms " + actionWordStem.lower() + "ed. Savefile name is: " + saveName)
print("Runtime was " + str(finish_time-start_time) + " s")

################################################################
################### 4. Prep for Monte Carlo ####################
################################################################
"""
# Decant the UM into a lighter package:
nb_terms = dec.decant_UM_terms(theUM)  # lists and numby-accelerated SigmoidCurves
stateImage = dec.readTerms(nb_terms) # holds UM parameters and max/min values
monte_carlo_syms = stitch.make_monte_carlo_lists(symTerms) # this will be
monte_carlo_syms.nb_terms = nb_terms
monte_carlo_syms.stateImage = stateImage

with open("_mc" + saveName, "wb") as f:
    pickle.dump(monte_carlo_syms, f)

del symMats # make sure it's not in the workspace, to free up memory

wrapper contains:
    # transfer to the wrapper
       # symmetryRowArray[termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
       # isSingleAtom[termNum]
    mcLists = MCwrapper()
    mcLists.callDist       # callDist[termNum][molPl][distortedBondNum][distortionNum] gives a numpy array
    mcLists.callFirstLast  # [termNum][molPl][distortedBondNum][distortionNum] --> [first0_last1,callNumber] 
                                                                                      # 2D numpy array

    mcLists.symmetryValArray 
    mcLists.symmetryIndsArray   # [termNum][molPl][distortedBondNum][distortionNum] 
                               # 2D numpy array of [row0_col1,indNumber]

    mcLists.isSingleAtom       # isSingleAtom[termNum] is boolean
    
    When isSingleAtom==True, the 'call...' lists are omitted and there are no 
    [distortedBondNum][distortionNum] indices
"""


mcLists = stitch.make_monte_carlo_lists(symMats)
mcLists.theUM = theUM
mcLists.numElectrons = [theUM.countElectrons(elList) for elList in structInfo.elementsLists]
mcLists.molNames = structInfo.molNames

saveName += '2'
with open(saveName, "wb") as f:
    pickle.dump(mcLists, f)




# configMats.UM=theUM  #***!!! remove this when the ConfigTerms object is updated

"""to do next:
    X1. Attempt to implement numba and run stitch more quickly
    2. add the ".makeHmats" method and test results for a simple molecule
       X-done for Coulomb terms
    X3. implement 'E' and 'pruneTermsList'
       X-for E: just create a full sparse matrix of ones (i.e. brute force!)
    X4. Implement a list of all variables in the UM, for convenience in the Monte Carlo stage
    5. Rename the ConfigTerms object (SymTerms?)
    
"""



