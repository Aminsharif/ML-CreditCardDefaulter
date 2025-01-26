import os
from ml_creditcard_defaulter import logger
import joblib
from ml_creditcard_defaulter.config.configuration import ModelPredictionConfig


class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config

    def find_correct_model_file(self,cluster_number):
            logger.info('Entered the find_correct_model_file method of the File_Operation class')
            try:
                folder_name=self.config.model_path
                list_of_files = os.listdir(folder_name)

                for file in list_of_files:
                    try:
                        if (file.index(str( cluster_number))!=-1):
                            model_name=file    
                    except:
                        continue
                model_name=model_name.split('.')[0]
                logger.info('Exited the find_correct_model_file method of the Model_Finder class.')
                return model_name
            except Exception as e:
                logger.info('Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(e))
                logger.info('Exited the find_correct_model_file method of the Model_Finder class with Failure')
                raise Exception()
            
    def load_model(self,filename):
        logger.info( 'Entered the load_model method of the File_Operation class')
        try:
            with open(self.config.model_path + '/' + filename + '.pkl' ,'rb') as f:
                logger.info('Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return joblib.load(f)
        except Exception as e:
            logger.info('Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str( e))
            logger.info('Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()
        
    