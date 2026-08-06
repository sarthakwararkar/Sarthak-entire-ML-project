import os 
import pandas as pd 
import numpy 
import sys

from dataclasses import dataclass

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils import evaluate_model,save_object

from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor ,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

@dataclass
class Modeltrainerconfig:
    trained_model_file_path = os.path.join('artifacts' , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = Modeltrainerconfig()

    def initiate_model_trainer(self , train_array , test_array ):
        try:
            logging.info("spliting training and test data")
            X_train  , y_train ,X_test , y_test = (
                train_array[: ,:-1],
                train_array[: ,-1],
                test_array[: ,:-1] ,
                test_array[:,-1],
            ) 
            models={
            "Random Forest":RandomForestRegressor(),
            "Decision tree":DecisionTreeRegressor(),
            "Gradient Boosting":GradientBoostingRegressor(),
            "Linear regression":LinearRegression() ,
            "K-Neighbours_regressor":KNeighborsRegressor(),
            "XGBOOST Regressor":XGBRegressor(),
            "CatBoostRegressor":CatBoostRegressor(verbose=False),
            "AdaBoostRegressor":AdaBoostRegressor(),     
            }

            model_report:dict=evaluate_model(X_train =X_train , y_train=y_train ,X_test =X_test ,y_test = y_test   ,models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6 :
                raise CustomException("no best model found")

            logging.info("best model found for both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted= best_model.predict(X_test)

            r2_square = r2_score(y_test , predicted)

            return r2_square

        except Exception as e :
            raise CustomException(e , sys)       
