import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        """ Train multiple regression models and select the best one based on R^2 score."""
        logging.info("Initiating model training")
        try:
            logging.info("Model training started")
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Support Vector Machine": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                 "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0)
            }

            model_scores = evaluate_model(
                X_train, X_test, y_train, y_test, models
            ) 
            
            best_model_score = max(model_scores.values())
            best_model_name = max(model_scores, key=model_scores.get)

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R^2 score >= 0.6")
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_score = best_model.score(X_test, y_test)

            return r2_score

        except Exception as e:
            raise CustomException(e)