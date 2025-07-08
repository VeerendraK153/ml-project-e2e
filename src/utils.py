import os
import sys
import pandas as pd
import dill
import numpy as np

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


