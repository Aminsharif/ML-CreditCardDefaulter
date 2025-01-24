import os
import pandas as pd
import numpy as np
import urllib.request as request
import zipfile
from ml_creditcard_defaulter import logger
from ml_creditcard_defaulter.utils.common import get_size
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from ml_creditcard_defaulter.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def remove_unwanted_spaces(self,data):
        data = pd.read_csv(self.config.data_path)

        try:
            df_without_spaces = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # drop the labels specified in the columns
            logger.info('Unwanted spaces removal Successful.Exited the remove_unwanted_spaces method of the Preprocessor class')
            return df_without_spaces
        except Exception as e:
            logger.info('Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()
    def remove_columns(self,data,columns):
        try:
            useful_data=data.drop(labels=columns, axis=1) # drop the labels specified in the columns
            logger.info('Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return useful_data
        except Exception as e:
            logger.info('Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            logger.info('Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()
        
    def separate_label_feature(self, data, label_column_name):
        try:
            X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            y=data[label_column_name] # Filter the Label columns
            logger.info('Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return X,y
        except Exception as e:
            logger.info('Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
        
    def is_null_present(self,data):
        null_present = False
        cols_with_missing_values=[]
        cols = data.columns
        try:
            null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])
            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(self.config.null_val_path) # storing the null column information to file
            logger.info('Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return null_present, cols_with_missing_values
        except Exception as e:
            logger.info('Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()
        
    def impute_missing_values(self, data, cols_with_missing_values):
        logger.info('Entered the impute_missing_values method of the Preprocessor class')
        
        cols_with_missing_values=cols_with_missing_values
        try:
            imputer = SimpleImputer(strategy="most_frequent")
            for col in cols_with_missing_values:
                data[col] = imputer.fit_transform(data[col])
            logger.info('Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data
        except Exception as e:
            logger.info('Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()
    def scale_numerical_columns(self,data):
        logger.info('Entered the scale_numerical_columns method of the Preprocessor class')

        try:
            num_df = data.select_dtypes(include=['int64']).copy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(num_df)
            scaled_num_df = pd.DataFrame(data=scaled_data, columns=num_df.columns)

            logger.info( 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return scaled_num_df

        except Exception as e:
            logger.info('Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info( 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()
    def encode_categorical_columns(self,data):
        logger.info( 'Entered the encode_categorical_columns method of the Preprocessor class')
        try:
            cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in cat_df.columns:
                cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)

            logger.info('encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return cat_df

        except Exception as e:
            logger.info('Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()
        
    def handle_imbalanced_dataset(self,x,y):
        logger.info('Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            rdsmple = RandomOverSampler()
            x_sampled,y_sampled  = rdsmple.fit_sample(x,y)
            logger.info('dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return x_sampled,y_sampled

        except Exception as e:
            logger.info('Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(e))
            logger.info('dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()
        
    def elbow_plot(self,data):
        logger.info( 'Entered the elbow_plot method of the KMeansClustering class')
        wcss=[] # initializing an empty list
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) # initializing the KMeans object
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show()
            plt.savefig('preprocessing_data/K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            logger.info( 'The optimum number of clusters is: '+str(kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return kn.knee

        except Exception as e:
            logger.info('Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            logger.info('Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

        
    def train_test_spliting(self, data):
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)