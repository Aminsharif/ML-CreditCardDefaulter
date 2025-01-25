import os
from ml_creditcard_defaulter import logger
from ml_creditcard_defaulter.components.data_transformation import DataTransformation
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics  import roc_auc_score,accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from ml_creditcard_defaulter.config.configuration import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def elbow_plot(self, data):

        logger.info('Entered the elbow_plot method of the KMeansClustering class')
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
            plt.savefig(self.config.elbow_png) # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            logger.info( 'The optimum number of clusters is: '+str(kn.knee)+' . Exited the elbow_plot method of the KMeansClustering class')
            return kn.knee

        except Exception as e:
            logger.info('Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  ' + str(e))
            logger.info('Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class')
            raise Exception()

    def create_clusters(self,data,number_of_clusters):
        logger.info( 'Entered the create_clusters method of the KMeansClustering class')
        try:
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            y_kmeans=kmeans.fit_predict(data) #  divide data into clusters

            joblib.dump(y_kmeans, os.path.join(self.config.root_dir, self.config.model_kmeans_name)) # saving the KMeans model to directory
                                                                                    # passing 'Model' as the functions need three parameters
            data['Cluster']=y_kmeans  # create a new column in dataset for storing the cluster information
            logger.info( 'succesfully created clusters. Exited the create_clusters method of the KMeansClustering class')
            return data
        except Exception as e:
            logger.info('Exception occured in create_clusters method of the KMeansClustering class. Exception message:  ' + str(e))
            logger.info('Fitting the data to clusters failed. Exited the create_clusters method of the KMeansClustering class')
            raise Exception()
        
    def get_best_params_for_naive_bayes(self,train_x,train_y):
        logger.info( 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            # param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}
            print(self.config.param_grid)
            param_grid = self.config.param_grid
            #Creating an object of the Grid Search class
            gnb =GaussianNB()
            grid = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=3,  verbose=3)
            #finding the best parameters
            grid.fit(train_x, train_y)

            #extracting the best parameters
            var_smoothing = grid.best_params_['var_smoothing']


            #creating a new model with the best parameters
            gnb = GaussianNB(var_smoothing=var_smoothing)
            # training the mew model
            gnb.fit(train_x, train_y)
            logger.info('Naive Bayes best params: '+str(grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return gnb
        except Exception as e:
            logger.info('Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(e))
            logger.info('Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()
        
    def get_best_params_for_xgboost(self,train_x,train_y):
        logger.info('Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            # param_grid_xgboost = {"n_estimators": [50,100, 130],"max_depth": range(3, 11, 1),"random_state":[0,50,100]}
            param_grid_xgboost = self.config.param_grid_xgboost
            # Creating an object of the Grid Search class
            grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),param_grid_xgboost, verbose=3,cv=2,n_jobs=-1)
            # finding the best parameters
            grid.fit(train_x, train_y)

            # extracting the best parameters
            random_state = grid.best_params_['random_state']
            max_depth = grid.best_params_['max_depth']
            n_estimators = grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            xgb = XGBClassifier(random_state=random_state, max_depth=max_depth,n_estimators= n_estimators, n_jobs=-1 )
            # training the mew model
            xgb.fit(train_x, train_y)
            logger.info('XGBoost best params: ' + str(
                                       grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return xgb
        except Exception as e:
            logger.info('Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(e))
            logger.info('XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()
        
    def get_best_model(self,train_x,train_y,test_x,test_y):

        logger.info('Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            prediction_xgboost =xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                xgboost_score = accuracy_score(test_y, prediction_xgboost)
                logger.info( 'Accuracy for XGBoost:' + str(xgboost_score))  # Log AUC
            else:
                xgboost_score = roc_auc_score(test_y, prediction_xgboost) # AUC for XGBoost
                logger.info( 'AUC for XGBoost:' + str(xgboost_score)) # Log AUC

            # create best model for Random Forest
            naive_bayes=self.get_best_params_for_naive_bayes(train_x,train_y)
            prediction_naive_bayes=naive_bayes.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                naive_bayes_score = accuracy_score(test_y,prediction_naive_bayes)
                logger.info( 'Accuracy for NB:' + str(naive_bayes_score))
            else:
                naive_bayes_score = roc_auc_score(test_y,prediction_naive_bayes) # AUC for Random Forest
                logger.info( 'AUC for RF:' + str(naive_bayes_score))

            #comparing the two models
            if(naive_bayes_score <  xgboost_score):
                return 'XGBoost',xgboost
            else:
                return 'NaiveBayes',naive_bayes

        except Exception as e:
            logger.info('Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(e))
            logger.info('Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
        
    
