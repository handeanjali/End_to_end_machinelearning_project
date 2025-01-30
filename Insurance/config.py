import pymongo
import pandas as pd
import numpy as np
import json
import os, sys
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("DATABASE_URL")

env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "charges"
print(env_var.mongo_db_url)