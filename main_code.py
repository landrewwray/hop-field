#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022


"""
import universalmodel.universalmodel as um
import load_create_mol as lcm 
import hmat_stitcher.make_mat_lists as mkLst
#1. code to load and create molecule --> molConfigList
atomsLists, bondsArrays, coordsArrays = lcm.load_mol.loadMol('filename') #path to all molecule data files
configsLists = lcm.config_wrapper('filename')
configs = lcm.config_wrapper.ConfigWrapper(configsLists, atomsLists, bondsArrays, coordsArrays)

#2. make the universal model
theUM = UniversalModel(['C','Cinit_0.txt'])
theUM += UniversalModel(['H','Hinit_0.txt'])  
theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

#3. create the matrices
configMats=mkLst.ConfigTerms(theUM,molConfigList)
configMats.save('trial1')
configMats.makeHmats(theUM)

#4. Now for Monte Carlo!

theVects=configMats.makeVects()
theParams=theUM.outputParams()
