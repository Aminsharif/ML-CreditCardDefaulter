from ml_creditcard_defaulter.config.configuration import ConfigurationManager
from ml_creditcard_defaulter.components.data_transformation import DataTransformation
import pandas as pd


STAGE_NAME = "Data validation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data = pd.read_csv(data_transformation_config.data_path)
        X, y = data_transformation.separate_label_feature(data,label_column_name='default payment next month')
        is_null_present,cols_with_missing_values=data_transformation.is_null_present(X)
        if(is_null_present):
            X=data_transformation.impute_missing_values(X,cols_with_missing_values)
        
        X['Labels']=y
        data_transformation.train_test_spliting(X)
