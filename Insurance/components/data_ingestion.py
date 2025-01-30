from Insurance import utils
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    """
    DataIngestion class handles the process of ingesting data from a source, 
    saving it into a feature store, and splitting it into training and testing datasets.
    """

    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        """
        Initializes DataIngestion with configuration details.
        
        Args:
            data_ingestion_config (config_entity.DataIngestionConfig): Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        """
        Executes the data ingestion process which includes fetching data, saving it to a feature store,
        and splitting it into training and testing datasets.

        Returns:
            artifact_entity.DataIngestionArtifact: Paths of the feature store, training, and testing datasets.
        """
        try:
            # Step 1: Export data from collection to a Pandas DataFrame
            logging.info("Exporting collection data as pandas dataframe")
            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )

            # Step 2: Replace missing values ("na") with NaN
            logging.info("Replacing 'na' with nan")
            df.replace(to_replace="na", value=np.nan, inplace=True)

            # Step 3: Save data in the feature store
            logging.info("Creating feature store directory if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("Saving dataframe to feature store")
            df.to_csv(
                path_or_buf=self.data_ingestion_config.feature_store_file_path,
                index=False,
                header=True
            )

            # Step 4: Split the dataset into training and testing sets
            logging.info("Splitting dataset into training and testing sets")
            train_df, test_df = train_test_split(
                df, 
                test_size=self.data_ingestion_config.test_size, 
                random_state=1
            )

            # Step 5: Save training and testing datasets
            logging.info("Creating dataset directory if not available")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("Saving training dataset")
            train_df.to_csv(
                path_or_buf=self.data_ingestion_config.train_file_path,
                index=False,
                header=True
            )

            logging.info("Saving testing dataset")
            test_df.to_csv(
                path_or_buf=self.data_ingestion_config.test_file_path,
                index=False,
                header=True
            )

            # Step 6: Prepare Data Ingestion Artifact
            logging.info("Preparing data ingestion artifact")
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(error_message=e, error_detail=sys)
