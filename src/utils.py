import os
import sys
import pandas as pd
import numpy as np 
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path ,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj , file_obj)
    except Exception as e :
        raise CustomException(e , sys)


def evaluate_model(X_train ,y_train , X_test , y_test , models , param):

    try:
        report={}

        for i in range(len(list(models))):
            model_key = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_key]

            gs = GridSearchCV(model , para , cv=3)
            gs.fit(X_train , y_train)

            logging.info("did hyper parameter training ")

            best_model = gs.best_estimator_
            models[model_key] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train ,y_train_pred)
            test_model_score = r2_score(y_test ,y_test_pred)    

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e :
        raise CustomException(e , sys)     
