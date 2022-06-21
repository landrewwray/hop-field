#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 07:03:08 2022


"""
import universalmodel.universalmodel as um
# import load_create_mol as lcm #***
# import hmat_stitcher.make_mat_lists as mkLst

#1. code to load and create molecule --> molConfigList


#2. make the universal model
theUM = um.UniversalModel(['C','Cinit_0.txt'])
theUM += um.UniversalModel(['H','Hinit_0.txt'])  
theUM.makeCrossTerms(['all'],'Hop_generic0.txt')

# #3. create the matrices
# configMats=mkLst.ConfigTerms(theUM,molConfigList)
# configMats.save('trial1')
# configMats.makeHmats(theUM)

# #4. Now for Monte Carlo!

# theVects=configMats.makeVects()
# theParams=theUM.outputParams()
