# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:36:11 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, '/mnt/c/Programacion/Data Science Projects/Fashion Sales/src')
from exception import CustomException
from logger import logging

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# =============================================================================================== #

# ======================================== Main classes ========================================= #
# This class will provide the paths for the inputs to the data transformation process
@dataclass
class DataTransformationConfig:
    # Preprocessor object filepath
    preprocessor_obj_file_path = os.path.join('data', 'preprocessor.pkl')
# =============================================================================================== #