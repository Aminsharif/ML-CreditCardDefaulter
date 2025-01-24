import os
import pandas as pd
import urllib.request as request
import zipfile
from ml_creditcard_defaulter import logger
from ml_creditcard_defaulter.utils.common import get_size
from ml_creditcard_defaulter.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_data(self):
        unzip_path = self.config.unzip_dir
        destination_path = os.path.join(unzip_path, self.config.file_name)
        os.makedirs(unzip_path, exist_ok=True)

        data = pd.read_excel(self.config.source_URL)
        data.to_csv(destination_path, index=False)
        logger.info(f"File successfully moved from {self.config.source_URL} to {destination_path}")
  
  