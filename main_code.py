#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022


"""
import universalmodel.universalmodel as um

import load_create_mol as lcm
import hmat_stitcher.hmat_stitcher as stitch

# 1. code to load and create molecule --> molConfigList
molFilePath="Molecular Structure Data/*.mol2"
bondsPerMol=5 # Number of bonds to distort in each molecule (if enough bonds exist)
atomsLists, bondsArrays, coordsArrays = lcm.load_mol.loadMol(molFilePath) 
configsLists, molNumList, distortedBonds = lcm.config_wrapper.allMolDistortions(molFilePath,bondsPerMol)
elementsLists = lcm.config_wrapper.elementsLists(atomsLists)
configs = lcm.config_wrapper.ConfigWrapper(configsLists, atomsLists, elementsLists, bondsArrays, coordsArrays) # distortLists, atomsLists, elementsLists, bondsArrays, coordsArrays

#2. make the universal model
theUM = um.UniversalModel(['C','Cinit_0.txt'])
theUM += um.UniversalModel(['H','Hinit_0.txt'])  
theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

# #3. create the matrices
configMats=stitch.ConfigTerms(theUM,configs)

# configMats.save('trial1')  #also need configMats.load(fileName)
# configMats.makeHmats(theUM)  #create the Hmats for a parthticular UM

# #4. Now for Monte Carlo!

# theVects=configMats.makeVects()
# theParams=theUM.outputParams()
