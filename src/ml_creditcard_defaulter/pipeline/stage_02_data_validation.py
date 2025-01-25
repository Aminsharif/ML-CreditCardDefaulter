from ml_creditcard_defaulter.config.configuration import ConfigurationManager
from ml_creditcard_defaulter.components.data_validation import DataValiadtion

STAGE_NAME = "Data validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns(data_validation_config.unzip_data_dir, data_validation_config.STATUS_FILE)
