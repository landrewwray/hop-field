#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022


"""
import universalmodel.universalmodel as um
# import load_create_mol as lcm #***
# import hmat_stitcher.hmat_stitcher as stitch

# 1. code to load and create molecule --> molConfigList
# atomsLists, bondsArrays, coordsArrays = lcm.load_mol.loadMol("C:/.../hop-field/Molecular Structure Data/*.mol2") 
# configsLists = lcm.config_wrapper.allMolDistortions("C:/.../hop-field/Molecular Structure Data/*.mol2")
# configs = lcm.config_wrapper.ConfigWrapper(configsLists, atomsLists, bondsArrays, coordsArrays)

#2. make the universal model
theUM = um.UniversalModel(['C','Cinit_0.txt'])
theUM += um.UniversalModel(['H','Hinit_0.txt'])  
theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

# #3. create the matrices
# configMats=stitch.ConfigTerms(theUM,molConfigList)
# configMats.save('trial1')  #also need configMats.load(fileName)
# configMats.makeHmats(theUM)  #create the Hmats for a parthticular UM

# #4. Now for Monte Carlo!

# theVects=configMats.makeVects()
# theParams=theUM.outputParams()
