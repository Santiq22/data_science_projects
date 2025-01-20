# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 23:16:53 2024

@author: Santiago Collazo
@original_author: Krish Naik
"""
# ========================================= Packages ============================================ #
import os
import sys
sys.path.insert(0, '/mnt/c/Programacion/Data Science Projects/Fashion Sales/src')
from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformation
# =============================================================================================== #

# ======================================== Main classes ========================================= #
@dataclass
class DataIngestionConfig:
    # Path to output datasets
    training_data_path: str = os.path.join('../data', 'training_data.csv')
    validation_data_path: str = os.path.join('../data', 'validation_data.csv')
    raw_data_path: str = os.path.join('../data', 'raw_data.csv')
    
class DataIngestion:
    def __init__(self):
        # This variable will consist in the input I need to initialize
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        # Here needs to be the code to read the data (from local, database, etc)
        logging.info("Entered the data ingestion method or component")
        
        try:
            # Here we need to specify the path to the data or access the database to get it
            df = pd.read_csv('../../notebook/data/mock_fashion_data_uk_us.csv')
            logging.info("Read the dataset as a pandas Data Frame object")
            
            # Create the output folders
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path), exist_ok = True)
            
            # Save the raw data in its path
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            # Splitting of the dataset into training and validation datasets
            logging.info("Training and validation split initiated")
            training_set, validation_set = train_test_split(df, test_size = 0.2, random_state = 10)
            
            # Save the training and validation data in the path specified above
            training_set.to_csv(self.ingestion_config.training_data_path, index = False, header = True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index = False, header = True)
            
            logging.info("Ingestion of the data completed")
            
            return(self.ingestion_config.training_data_path, self.ingestion_config.validation_data_path)
        except Exception as e:
            raise CustomException(e, sys)
# =============================================================================================== #

if __name__ == "__main__":
    obj = DataIngestion()
    training_data, validation_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    t_arr, v_arr, _ = data_transformation.initiate_data_transformation(training_data, validation_data)