import os
import sys
import pandas as pd
import dill
import numpy as np
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object):
    """
    Saves an object to a file using dill.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj (object): The object to be saved.
    """
    try:
        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved successfully")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        model_scores = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_scores[model_name] = test_model_score
            logging.info(f"{model_name} - Train R^2 Score: {train_model_score}, Test R^2 Score: {test_model_score}")
        return model_scores
    except Exception as e:
        raise CustomException(e, sys)