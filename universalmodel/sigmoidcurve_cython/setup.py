#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 12:05:08 2022

@author: lawray
"""
from distutils.core import setup
from Cython.Build import cythonize

# setup(ext_modules = cythonize('sigmoidcurve_c.pyx'))

setup(
      ext_modules = cythonize('test_tst1.pyx'),
      zip_safe=False,
)