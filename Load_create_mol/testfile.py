# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:39:06 2022

@author: danie
"""

import numpy as np
import pandas as pd
from math import floor
from os import listdir
import copy
import Load_mol as lm
import distort_mol as dm
import config_wrapper as cw
import random

atoms_list, coords_arry, bonds_arry = lm.loadFile("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/(5)Helicene(C22H14).mol2")
atomsLists, bondsArrays, coordsArrays = lm.loadMol("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/*.mol2")

distortList = cw.make1MolDistortions(atomsLists[0], coordsArrays[0], bondsArrays[0], 10, random.sample(
            list(range(bonds_arry.shape[0])), 10
        ))


distortionsLists = cw.allMolDistortions("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/*.mol2")
# newCoordsList = dm.distortionList(coords_arry, bonds_arry, np.arange(len(atoms_list)))