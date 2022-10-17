#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:05:08 2022

@author: lawray
"""
import sigmoidcurve as sc
import numpy as np
from numba import jit
from numba import float64

import time

spec = [
    ('theVal', float64),            
]

@jit
def runFast():
    tmp=sc.SigmoidCurve(30,0.2,5)
    tmp.initAmpsLinear(5)
    theVal=0.
    for pl in range(10000):
        theVal+=tmp.readVal(np.random.rand()*5)
    
    return theVal
    

start_time=time.time()

outVal=runFast()

end_time=time.time()
print('Runtime: ' + str(end_time-start_time))

start_time=time.time()

outVal=runFast()

end_time=time.time()
print('Runtime: ' + str(end_time-start_time))

print(outVal)

