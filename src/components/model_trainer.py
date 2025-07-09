import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils import save_object, evaluate_model_cv, tune_with_random_search, tune_with_optuna
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training started")

            models_random = {
                "Decision Tree": (DecisionTreeRegressor(), {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }),
                "SVR": (SVR(), {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                }),
                "K-Neighbors": (KNeighborsRegressor(), {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                }),
                "AdaBoost": (AdaBoostRegressor(), {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                }),
                "Linear Regression": (LinearRegression(), {})  # no tuning
            }

            optuna_models = {
                "Random Forest": (RandomForestRegressor, self.rf_param_space),
                "Gradient Boosting": (GradientBoostingRegressor, self.gb_param_space),
                "XGBoost": (XGBRegressor, self.xgb_param_space),
                "CatBoost": (CatBoostRegressor, self.cb_param_space)
            }

            best_models = {}
            scores = {}

            for name, (model, params) in models_random.items():
                logging.info(f"Tuning {name}")
                if params:
                    tuned_model = tune_with_random_search(model, params, X_train, y_train)
                else:
                    tuned_model = model
                    tuned_model.fit(X_train, y_train)
                cv_score = evaluate_model_cv(tuned_model, X_train, y_train)
                best_models[name] = tuned_model
                scores[name] = cv_score
                logging.info(f"{name} CV R2 score: {cv_score}")

            for name, (model_class, param_func) in optuna_models.items():
                logging.info(f"Optuna tuning for {name}")
                tuned_model, cv_score = tune_with_optuna(name, model_class, param_func, X_train, y_train)
                best_models[name] = tuned_model
                scores[name] = cv_score
                logging.info(f"{name} Optuna CV R2 score: {cv_score}")

            # Select best model based on CV score
            best_model_name = max(scores, key=scores.get)
            best_model = best_models[best_model_name]
            best_cv_score = scores[best_model_name]
            logging.info(f"Best model (CV): {best_model_name} with R2: {best_cv_score}")

            # Evaluate on final test set
            test_r2 = best_model.score(X_test, y_test)
            logging.info(f"Test R2 score for best model: {test_r2}")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")

            return test_r2

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def rf_param_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }

    @staticmethod
    def gb_param_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }

    @staticmethod
    def xgb_param_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }

    @staticmethod
    def cb_param_space(trial):
        return {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        }
