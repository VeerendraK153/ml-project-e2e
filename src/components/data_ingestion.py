import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_preprocessing import DataPreprocessing, DataPreprocessingConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')  

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            if not os.path.exists('notebook/data/train.csv'):
                logging.error("File not found: notebook/data/train.csv")
                raise CustomException("Input file not found", sys)

            df = pd.read_csv('notebook/data/train.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and Test split completed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train and Test data saved to artifacts folder")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    logging.info(f"Data ingestion completed successfully. Train path: {train_path}, Test path: {test_path}")
    data_preprocessing_obj = DataPreprocessing()
    X_train, X_test, y_train, y_test,preprocessor_path = data_preprocessing_obj.initiate_data_preprocessing(train_path, test_path)
    print(X_train.shape, X_test.shape,preprocessor_path)

    model_trainer_obj = ModelTrainer()
    best_score=model_trainer_obj.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print(f"Best model score: {best_score}")



