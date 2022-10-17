#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:05:08 2022

@author: lawray
"""
import numpy as np
from matplotlib import pyplot as plt
from math import floor
from numba.experimental import jitclass
from numba import int32, float32, float64

import time

spec = [
    ('sNum', int32),            
    ('dMin', float64),          
    ('dMax', float64),
    ('I_of_r', float64[:]),
    ('sAmps', float64[:]),
    ('_sWidth', float64),
    ('_sStart', float64[:]),
    ('Ix', float64[:]),
    ('Imax', float64[:]),
    ('Imin', float64[:]),
]


cpdef class SigmoidCurve:
    """Class storing a radially-resolved Hamiltonian parameter, framed as the sum of 
    2-parabola sigmoid-like functions.  The functions are defined as:
      -- for the first 1/2 of self._sWidth: 0.5*self.sAmps[pl]*((self._sStart[pl]-dVal)/(self._sWidth/2))**2
      -- for the next 1/2: self.sAmps[pl]*(1 - 0.5*((self._sStart[pl]-self._sWidth-dVal)/(self._sWidth/2))**2)
      -- the curve is flat outside this range
    Sigmoid functions have a 50% overlap with each of their nearest neighbors, 
    so that setting self.sAmps[:]=const gives a constant slope for the 
    Hamiltonian parameter
    
    Monte Carlo operations on this list should include:
    1. Adding Val to self.sAmps[pl] while subtracting Val from self.Samp[pl+1] (if pl+1 index is valid)
        -this allows for fine tuning of the large d parameters, while keeping small-d values fixed
    2. Same as (1), but subtract Val/N from pl+1...pl+N
    3. Alternatively, one can use I_of_r and add to 2 adjacent values at the same time to create a clean
       local distortion.
    
    Member functions are:
        readVal
        plot
        initAmpsLinear
        initRanges
        make_I_of_r
        make_sAmps
    
    Whenever updating self.I_of_r, you must run self.make_sAmps() to enable self.readVal(dist). Likewise, 
    self.make_I_of_r() should be run after updating sAmps if you would like to use the I_of_r representation
    
    reason is partly that self.readVal only
    references self.sAmp, and partly that self.sAmp is assumed to be the standard 
    
    :ivar sNum: Number of sigmoids defining a curve
    :ivar dMin: Minimum radial distance - the curve is flat beyond this point
    :ivar dMax: Maximum radial distance - the curve is 0 beyond this point
    
    """

    def __init__(self, numSigmoids, dMin=0.5, dMax=6.):
        self.sNum = numSigmoids
        self.dMin = dMin
        self.dMax = dMax
        self.I_of_r = np.zeros(self.sNum)
        self.sAmps = np.zeros(self.sNum)
        # self._maintain_I_of_r = True # update I_of_r when updating sAmp ***IMPLEMENT***

        # width of each sigmoid; place the near-field start in the middle of the last sigmoid:
        self._sWidth = 2 * (dMax - dMin) / numSigmoids

        # a new sigmoid starts after each half-width:
        self._sStart = dMax - np.arange(numSigmoids) * (self._sWidth / 2)
        self.Ix = self.dMax - (1 + np.arange(numSigmoids)) * (self._sWidth / 2)

    def initAmpsLinear(self, nearFieldVal):
        # init sigmoid amplitudes:
        self.sAmps = np.ones(self.sNum) * nearFieldVal / (self.sNum - 0.5)

        # define the alternative representations in terms of curve amplitude at the sigmoid centroids
        self.make_I_of_r()

    def initRanges(self, allMin=-0.5, allMax=2, absRange=0):
        """Define the allowed range of each parameter, for Monte Carlo sampling
        """

        if absRange:
            self.max_sAmps = allMax * np.ones(self.sNum)
            self.min_sAmps = allMin * np.ones(self.sNum)
        else:
            self.max_sAmps = allMax * (self.readVal(0) / self.sNum) * np.ones(self.sNum)
            self.min_sAmps = allMin * (self.readVal(0) / self.sNum) * np.ones(self.sNum)

    def init_I_limits(self, theRange):
        if self.I_of_r[-1] == 0:  # create linearly shrinking bounds
            self.Imax = theRange[1] * (np.arange(self.sNum) + 0.5) / (self.sNum - 0.5)
            self.Imin = theRange[0] * (np.arange(self.sNum) + 0.5) / (self.sNum - 0.5)
        else:  # define the bounds by rescaling I_of_r
            self.Imax = self.I_of_r * (theRange[1] / self.I_of_r[-1])
            self.Imin = self.I_of_r * (theRange[0] / self.I_of_r[-1])

    def make_I_of_r(self):  ####*****test
        """
        Create the curve contour self.I_of_r from sigmoid amplitudes self.sAmps.
        If one wishes to work with a position representation, one must first call
        self.make_I_of_r, and then call self.make_sAmps every time self.I_of_r is updated
        
        
        """

        self.I_of_r[0] = self.sAmps[0] / 2
        for pl in range(1, self.sNum):
            # each self.I_of_r increment is self.sAmps[pl-1]/2 + self.sAmps[pl]/2
            self.I_of_r[pl] = (
                self.I_of_r[pl - 1] + self.sAmps[pl - 1] / 2 + self.sAmps[pl] / 2
            )

    def make_sAmps(self):  ####*****test
        """Create sigmoid amplitudes (self.sAmps), given curve intensities at the sigmoid centroids

        :iVar I_of_r:  
        """

        self.sAmps[0] = 2 * self.I_of_r[0]
        for pl in range(1, self.sNum):
            # each self.I_of_r increment is self.sAmps[pl-1]/2 + self.sAmps[pl]/2
            self.sAmps[pl] = 2 * (
                self.I_of_r[pl] - self.I_of_r[pl - 1] - self.sAmps[pl - 1] / 2
            )

    def readVal(self, dist):
        """Output the curve amplitude for a given bond distance. Note that readVal
        references self.sAmps, so self.make_sAmps() must be called first if self.I_of_r
        is the vector being manipulated.
        
        :param dist: Distance from an atom.
        """

        # first set out-of-bounds values to the outermost in-bounds values
        if dist <= self.dMin:
            maxVal = np.sum(self.sAmps[:-1]) + 0.5 * self.sAmps[-1]
            return maxVal
        elif dist >= self.dMax:
            return 0

        # now deal with in-bounds cases
        else:
            # first frame the distance from the far-field, in units of _sWidth:
            normDist = (self.dMax - dist) / self._sWidth

            # value contribution from full sigmoids:
            fullSigmoids = floor(2 * normDist - 1)
            fullSigmoids = fullSigmoids * (fullSigmoids > 0)
            # disp(fullSigmoids)
            # self.fullSigmoids=fullSigmoids
            theVal = np.sum(self.sAmps[:fullSigmoids])

            # now add the partial sigmoid component:
            if normDist > 0.5:  # if we're considering the overlap of 2 sigmoids:
                # define partial dist in units of self._sWidth/2
                partialDist = 2 * (normDist - (fullSigmoids + 1) / 2)
                # print(partialDist)
                # up parabola
                theVal += 0.5 * self.sAmps[fullSigmoids + 1] * partialDist ** 2

                # down parabola
                theVal += self.sAmps[fullSigmoids] * (1 - 0.5 * (1 - partialDist) ** 2)
            else:
                partialDist = 2 * normDist
                # print(partialDist)
                # up parabola only
                theVal += 0.5 * self.sAmps[fullSigmoids] * partialDist ** 2

            return theVal

    def plot(self, numPoints=50):
        """Plot the full curve stored in SigmoidCurve
        """

        totalRange = self.dMax - self.dMin  # * (self.sNum+1)/self.sNum
        firstVal = self.dMin  # -(self.dMax - self.dMin)*1/self.sNum
        xVals = firstVal + totalRange * np.asarray(range(numPoints)) / (numPoints - 1)

        theCurve = [self.readVal(thePl) for thePl in xVals]
        plt.plot(xVals, theCurve)


# start_time=time.time()

# tmp=SigmoidCurve(30)
# tmp.initAmpsLinear(5)
# theVal=0.
# for pl in range(10000):
#     theVal=tmp.readVal(np.random.rand()*5)

# end_time=time.time()
# print('Runtime: ' + str(end_time-start_time))

