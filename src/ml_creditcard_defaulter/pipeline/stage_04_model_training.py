import os
from ml_creditcard_defaulter.config.configuration import ConfigurationManager
from ml_creditcard_defaulter.components.data_transformation import DataTransformation
from ml_creditcard_defaulter.components.model_training import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from ml_creditcard_defaulter import logger

STAGE_NAME = "Model training stage"

class DataModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_trainer_config()
        model_training = ModelTrainer(config=model_training_config)
        data_transformation = DataTransformation(config=model_training_config)
        data = pd.read_csv(model_training_config.train_data_path)

        X = data.drop(['Labels'], axis=1)
        number_of_clusters = model_training.elbow_plot(X)
        X = model_training.create_clusters(X, number_of_clusters)
        
        X['Labels'] = data['Labels']
        list_of_clusters=X['Cluster'].unique()
        for i in list_of_clusters:
            cluster_data=X[X['Cluster']==i] # filter the data for one cluster

                    # Prepare the feature and Label columns
            cluster_features=cluster_data.drop(['Labels','Cluster'],axis=1)
            cluster_label= cluster_data['Labels']

                    # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
                    # Proceeding with more data pre-processing steps
            train_x = data_transformation.scale_numerical_columns(x_train)
            test_x = data_transformation.scale_numerical_columns(x_test)
            best_model_name,best_model=model_training.get_best_model(train_x,y_train,test_x,y_test)
            print('......................................' )
            joblib.dump(best_model, os.path.join(model_training_config.model_dir, best_model_name+str(i)+'.pkl'))
            logger.info(f'Successful End of Training with best model {best_model_name+str(i)}')
