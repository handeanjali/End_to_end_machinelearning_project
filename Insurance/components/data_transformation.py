import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from Insurance import utils
from Insurance.config import TARGET_COLUMN

class DataTransformation:
    """
    Handles data transformation including missing values imputation,
    outlier handling, categorical encoding, and feature scaling.
    """
    
    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initializes DataTransformation with configuration and ingestion artifacts.
        """
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
    
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates and returns a transformation pipeline consisting of an imputer and scaler.
        """
        try:
            pipeline = Pipeline([
                ('Imputer', SimpleImputer(strategy='constant', fill_value=0)),
                ('Scaler', RobustScaler())
            ])
            return pipeline
        except Exception as e:
            raise InsuranceException(e, sys)
    
    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        """
        Transforms the dataset by handling missing values, encoding categorical variables,
        and scaling numerical features.
        """
        try:
            # Load training and test datasets
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Separate features and target variable
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]
            
            # Encode categorical features
            label_encoder = LabelEncoder()
            for col in X_train.select_dtypes(include=['object']).columns:
                X_train[col] = label_encoder.fit_transform(X_train[col])
                X_test[col] = label_encoder.transform(X_test[col])
            
            # Apply transformation pipeline
            transformer = self.get_data_transformer_object()
            transformer.fit(X_train)
            X_train_transformed = transformer.transform(X_train)
            X_test_transformed = transformer.transform(X_test)
            
            # Combine transformed features with target variable
            train_array = np.c_[X_train_transformed, y_train.values]
            test_array = np.c_[X_test_transformed, y_test.values]
            
            # Save transformed data and objects
            utils.save_numpy_array_data(self.data_transformation_config.transformed_train_path, train_array)
            utils.save_numpy_array_data(self.data_transformation_config.transformed_test_path, test_array)
            utils.save_object(self.data_transformation_config.transform_object_path, transformer)
            utils.save_object(self.data_transformation_config.target_encoder_path, label_encoder)
            
            # Return transformation artifact
            return artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )
        except Exception as e:
            raise InsuranceException(e, sys)
