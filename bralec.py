# -*- coding: utf-8 -*-
"""
Created on Fri May 13 10:13:03 2016

@author: matevzk
"""
import configparser
from configparser import ConfigParser
import logging

mainConfigPath = 'genetic.ini'
mainAlgConfigPath = 'algorithm.ini'



logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.WARNING)


# Genetic algorithm configuration       
GE_cf = ConfigParser()
GE_cf.read(mainConfigPath)

# Matrix factorization configuration
alg_cf = ConfigParser()
alg_cf.read(mainAlgConfigPath)


