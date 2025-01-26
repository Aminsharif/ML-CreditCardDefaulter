import os
from ml_creditcard_defaulter.config.configuration import ConfigurationManager
from ml_creditcard_defaulter.components.data_transformation import DataTransformation
from ml_creditcard_defaulter.components.model_training import ModelTrainer
import pandas as pd
import joblib
from ml_creditcard_defaulter import logger
from ml_creditcard_defaulter.components.prediction import ModelPrediction

STAGE_NAME = "Model training stage"

class PredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                model_prediction_config = config.get_model_predict_config()
                model_prediction = ModelPrediction(config=model_prediction_config)
                
                data = pd.read_csv(model_prediction_config.predict_default_data_path)
                
                is_null_present,cols_with_missing_values=data_transformation.is_null_present(data)
                
                if(is_null_present):
                        data=data_transformation.impute_missing_values(data,cols_with_missing_values)

                X = data_transformation.scale_numerical_columns(data)

                kmeans_model = joblib.load(model_prediction_config.predict_kmeans_model_path)

                clusters=kmeans_model.predict(X)#drops the first column for cluster prediction
                
                X['clusters']=clusters
                clusters=X['clusters'].unique()
                for i in clusters:
                        cluster_data= X[X['clusters']==i]
                        cluster_data = cluster_data.drop(['clusters'],axis=1)
                        model_name = model_prediction.find_correct_model_file(i)
                        model = model_prediction.load_model(model_name)
                        result=(model.predict(cluster_data))

                        final= pd.DataFrame(list(zip(result)),columns=['Predictions'])
                        path= os.path.join(model_prediction_config.prediction_output, 'prediction.csv')
                        final.to_csv(path, header=True,mode='a+') #appends result to prediction file
                        logger.info('End of Prediction')
        except Exception as ex:
                logger.info( 'Error occured while running the prediction!! Error:: %s' % ex)
                raise ex
