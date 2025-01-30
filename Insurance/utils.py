import pandas as pd
import numpy as np
import os
import sys
import yaml
import dill
from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.config import mongo_client

# -----------------------------------------------
# Fetch Collection as DataFrame
# -----------------------------------------------
def get_collection_as_dataframe(database_name: str, collection_name: str) -> pd.DataFrame:
    """
    Retrieves a collection from MongoDB and returns it as a Pandas DataFrame.
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info("Dropping column _id")
            df = df.drop("_id", axis=1)
        logging.info(f"Rows and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise InsuranceException(e, sys)

# -----------------------------------------------
# Write YAML File
# -----------------------------------------------
def write_yaml_file(file_path: str, data: dict):
    """
    Writes data to a YAML file.
    """
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise InsuranceException(e, sys)

# -----------------------------------------------
# Convert DataFrame Columns to Float
# -----------------------------------------------
def convert_columns_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Converts numerical columns in the DataFrame to float, excluding specified columns.
    """
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if df[column].dtypes != 'O':
                    df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise e

# -----------------------------------------------
# Save Object to File
# -----------------------------------------------
def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise InsuranceException(e, sys) from e

# -----------------------------------------------
# Load Object from File
# -----------------------------------------------
def load_object(file_path: str) -> object:
    """
    Loads an object from a file using dill.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys) from e

# -----------------------------------------------
# Save NumPy Array to File
# -----------------------------------------------
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Saves a NumPy array to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise InsuranceException(e, sys) from e


#***********************************## Model Training*******************************************

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys) from e