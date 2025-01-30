from Insurance.entity import artifact_entity,config_entity
from Insurance.entity.config_entity import DataValidationConfig

from Insurance.exception import InsuranceException
from Insurance.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os,sys 
import pandas as pd
from Insurance import utils
import numpy as np
from Insurance.config import TARGET_COLUMN


class DataValidation:
    """
    This class handles data validation by checking for missing values,
    ensuring required columns exist, and detecting data drift.
    """

    def __init__(self,
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise InsuranceException(e,sys)
        
    # Drop columns with missing values above a threshold
    def drop_missing_values_column(self, df:pd.DataFrame, report_key_name: str) -> Optional[pd.DataFrame]:
           """
        Drops columns where missing values exceed the threshold defined in config.
        """

           try:
                threshold = self.data_validation_config.missing_threshold
                null_report = df.isna().sum()/df.shape[0]

                #selecting column name which contains null
                logging.info(f"selecting column name which contains null above to {threshold}")

                drop_column_names = null_report[null_report > threshold].index

                logging.info(f"Columns to drop: {list(drop_column_names)}")

                self.validation_error[report_key_name]=list(drop_column_names)
                df.drop(list(drop_column_names),axis=1, inplace=True)
                return df if len(df.columns) > 0 else None
           except Exception as e:
               raise InsuranceException(e, sys)
           
    # Check if all required columns exist in the dataset
    def is_required_column_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
         """
        Ensures that all required columns in the base dataset exist in the current dataset.
        """
         try:
              missing_columns = [col for col in base_df.columns if col not in current_df.columns]
              if missing_columns:
                   logging.info(f"Missing columns: {missing_columns}")
                   self.validation_error[report_key_name] = missing_columns
                   return False
              return True
         except Exception as e:
              raise InsuranceException(e,sys)
        

    # Detect data drift using KS test
    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
         """
        Detects data drift using the Kolmogorov-Smirnov (KS) test between base and current datasets.
        """
         try:
              drift_report = {}
              for column in base_df.columns:
                   base_data, current_data = base_df[column], current_df[column]
                   same_distribution = ks_2samp(base_data, current_data)
                   drift_report[column] = {
                                            "pvalue": float(same_distribution.pvalue),
                                            "same_distribution":same_distribution.pvalue > 0.05
                                            }
                   self.validation_error[report_key_name] = drift_report
         except Exception as e:
              raise InsuranceException(e,sys)
         

     # Initiate data validation process
    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        """
        Executes the complete data validation process.
        """
        try:
             logging.info("Reading base dataset")
             base_df = pd.read_csv(self.data_validation_config.base_file_path)
             base_df.replace({"na":np.nan}, inplace= True)

             logging.info("Dropping columns with excessive missing values in base dataset")
             base_df = self.drop_missing_values_column(base_df, "missing_values_within_base_dataset")

             logging.info("Reading training and test datasets")
             train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
             test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

             train_df = self.drop_missing_values_column(train_df,"missing_values_within_train_dataset")
             test_df = self.drop_missing_values_column(test_df, "missing_values_within_test_dataset")

             exclude_columns = [TARGET_COLUMN]
             base_df = utils.convert_columns_float(base_df, exclude_columns)
             train_df = utils.convert_columns_float(train_df, exclude_columns)
             test_df = utils.convert_columns_float(test_df, exclude_columns)

             logging.info("Checking for required columns in train dataset")
             train_columns_status = self.is_required_column_exists(base_df, train_df, "missing_columns_within_train_dataset")

             logging.info("Checking for required columns in test dataset")
             test_columns_status = self.is_required_column_exists(base_df, test_df, "missing_columns_within_test_dataset")
        
             if train_columns_status:
                logging.info("Detecting data drift in train dataset")
                self.data_drift(base_df, train_df, "data_drift_within_train_dataset")
            
             if test_columns_status:
                logging.info("Detecting data drift in test dataset")
                self.data_drift(base_df, test_df, "data_drift_within_test_dataset")
            
             logging.info("Writing validation report to YAML file")
             utils.write_yaml_file(self.data_validation_config.report_file_path, self.validation_error)
            
             data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
             logging.info(f"Data validation artifact created: {data_validation_artifact}")
             return data_validation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

         
                     


