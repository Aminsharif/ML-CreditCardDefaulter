import os
import urllib.request as request
import zipfile
from ml_creditcard_defaulter import logger
from ml_creditcard_defaulter.utils.common import get_size
from ml_creditcard_defaulter.config.configuration import DataValidationConfig
import pandas as pd

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self, file_path, stutus_file)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(file_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(stutus_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(stutus_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

  