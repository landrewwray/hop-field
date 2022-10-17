#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:05:08 2022

@author: lawray
"""


#1. test accessability of cdef member variables
#2. test array/list input into double[2]


cpdef class DummyClass:
    cdef double a
    
    cdef __init__(self, double inputVar):
        self.a = inputVar
    
    cdef double doubleVar()
        return self.a * 2
    
    