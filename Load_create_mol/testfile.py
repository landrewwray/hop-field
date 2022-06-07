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

atoms_list, coords_arry, bonds_arry = lm.loadFile("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/(5)Helicene(C22H14).mol2")
moleculesList = lm.loadMol("C:/Users/danie/OneDrive/Documents/GitHub/hop-field/Molecular Structure Data/*.mol2")
