# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:26:46 2022

@author: danie
"""

import load_create_mol as lcm
# from lcm import load_mol, distort_mol, config_wrapper

atomsLists, bondsArrays, coordsArrays = lcm.load_mol.loadMol("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/*.mol2") #path specific to device?
configsLists = lcm.config_wrapper.allMolDistortions("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/*.mol2")
configs = lcm.config_wrapper.ConfigWrapper(configsLists, atomsLists, bondsArrays, coordsArrays)