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
from utils import save_object

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
    
# Class to set the inputs
class DataTransformation:
    def __init__(self):
        # Attribute representing the preprocessor object filepath
        self.data_transformation_config = DataTransformationConfig()
    
    # This function creates the data transformer object based on different types of data
    def get_data_transformer_object(self):
        try:
            # Separate between different types of predictors
            numerical_columns = ['Rating', 'Review Count', 'Age']
            categorical_columns = ['Brand', 
                                   'Category', 
                                   'Description', 
                                   'Style Attributes', 
                                   'Total Sizes', 
                                   'Available Sizes', 
                                   'Color', 
                                   'Purchase History', 
                                   'Fashion Magazines', 
                                   'Fashion Influencers', 
                                   'Season', 
                                   'Time Period Highest Purchase', 
                                   'Customer Reviews', 
                                   'Social Media Comments', 
                                   'feedback']
            
            # Pipeline to apply to training numerical predictors
            numerical_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy = "median")),
                                                   ("scaler", StandardScaler(with_mean = False))])
            
            logging.info("Numerical columns transformation completed")
            
            # Pipeline to apply to training categorical predictors
            categorical_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy = "most_frequent")),
                                                     ("one_hot_encoder", OneHotEncoder()),
                                                     ("scaler", StandardScaler(with_mean = False))])
            
            logging.info("Categorical columns transformation completed")
            
            # Process of combining both pipelines
            preprocessor = ColumnTransformer([("numerical_pipeline", numerical_pipeline, numerical_columns),
                                              ("categorical_pipeline", categorical_pipeline, categorical_columns)])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    # This function initiates the data transformation process
    def initiate_data_transformation(self, training_path, validation_path):
        try:
            # Load training dataset
            training_df = pd.read_csv(training_path)
            
            # Load validation dataset
            validation_df = pd.read_csv(validation_path)
            
            logging.info("Reading of training and validation data completed")
            
            logging.info("Obtaining preprocessing object")
            
            # Data transformer object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Response variable
            target_column_name = "Price"
            
            # Numerical predictors
            numerical_columns = ['Rating', 'Review Count', 'Age']
            
            # Training predictors
            input_feature_training_df = training_df.drop(columns = [target_column_name], axis = 1)
            
            # Training response
            target_feature_training_df = training_df[target_column_name]
            
            # Validation predictors
            input_feature_validation_df = validation_df.drop(columns = [target_column_name], axis = 1)
            
            # Validation response
            target_feature_validation_df = validation_df[target_column_name]
            
            logging.info("Applying preprocessing object on training data and validation data")
            
            # Fit and transform the training predictors
            input_feature_training_arr = preprocessing_obj.fit_transform(input_feature_training_df)
            
            # Transform the validation predictors
            input_feature_validation_arr = preprocessing_obj.transform(input_feature_validation_df)
            
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(input_feature_training_arr.shape)
            print(input_feature_validation_arr.shape)
            print(np.array(target_feature_training_df).shape)
            print(np.array(target_feature_validation_df).shape)
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            
            """ Here it is not working """
            # Concatenate training arrays along their second axis
            training_arr = np.c_[input_feature_training_arr, np.array(target_feature_training_df)]
            
            # Concatenate validation arrays along their second axis
            validation_arr = np.c_[input_feature_validation_arr, np.array(target_feature_validation_df)]
            
            logging.info("Preprocessing object saved")
            
            # Save the preprocessor as a pickle file
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj)
            
            return(training_arr, validation_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #