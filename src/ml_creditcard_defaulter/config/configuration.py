from ml_creditcard_defaulter.constants import *
from ml_creditcard_defaulter.utils.common import read_yaml, create_directories
from ml_creditcard_defaulter.entity.config_entity import (DataIngestionConfig, 
                                                          DataValidationConfig, 
                                                          DataTransformationConfig,
                                                          ModelTrainerConfig,
                                                          ModelPredictionConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir,
                file_name=config.file_name
            )

        return data_ingestion_config
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.ColName

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            null_val_path = config.null_val_path,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params
        params.param_grid["var_smoothing"] = [float(v) for v in params.param_grid["var_smoothing"]]

        create_directories([config.root_dir, config.model_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            model_kmeans_name = config.model_kmeans_name,
            model_xabost_name = config.model_xabost_name,
            elbow_png=config.elbow_png,
            param_grid=params.param_grid,
            param_grid_xgboost=params.param_grid_xgboost,
            model_dir=config.model_dir,
        )

        return model_trainer_config
    
    def get_model_predict_config(self) -> ModelPredictionConfig:
        config = self.config.model_prediction
        schema = self.schema.ColName
        
        create_directories([config.root_dir, config.prediction_output])

        model_predict_config = ModelPredictionConfig(
            root_dir=config.root_dir,
            predict_default_data_path=config.predict_default_data_path,
            predict_data_path = config.predict_data_path,
            predict_validation_status_file = config.predict_validation_status_file,
            all_schema=schema,
            predict_kmeans_model_path = config.predict_kmeans_model_path,
            model_path = config.model_path,
            prediction_output = config.prediction_output,
        )
        return model_predict_config