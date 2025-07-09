import sys
import dill
import numpy as np
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

import pickle

from src.exception import CustomException
from src.logger import logging

def save_object(file_path: str, obj: object):
    try:
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model_cv(model, X_train, y_train, cv=5):
    """Evaluate model using cross-validation mean R2 score."""
    try:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        mean_score = np.mean(scores)
        return mean_score
    except Exception as e:
        raise CustomException(e, sys)

def tune_with_random_search(model, param_distributions, X_train, y_train, cv=5, n_iter=20):
    """RandomizedSearchCV tuning with CV score."""
    try:
        rs = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, cv=cv, scoring='r2', n_jobs=-1, random_state=42)
        rs.fit(X_train, y_train)
        logging.info(f"Best params (RandomSearch): {rs.best_params_}")
        return rs.best_estimator_
    except Exception as e:
        raise CustomException(e, sys)

def tune_with_optuna(model_name, model_class, param_space_func, X_train, y_train, cv=5, n_trials=20):
    """Optuna tuning with CV score."""
    def objective(trial):
        params = param_space_func(trial)
        model = model_class(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logging.info(f"Best params (Optuna) for {model_name}: {study.best_params}")

    best_model = model_class(**study.best_params)
    best_model.fit(X_train, y_train)
    return best_model, study.best_value

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
