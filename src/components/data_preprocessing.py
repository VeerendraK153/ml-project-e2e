import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataPreprocessingConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()

    def feature_engineering(self, df):
        """Perform feature engineering like creating age features, dropping columns, etc."""
        try:
            logging.info("Starting feature engineering")
            
            df = df.copy()
            
            df['MSSubClass'] = df['MSSubClass'].astype('str')
            df['age'] = df['YrSold'] - df['YearBuilt']
            df['age_mod'] = df['YearRemodAdd'] - df['YearBuilt']
            df['garage_age'] = df['YrSold'] - df['GarageYrBlt']
            
            df.drop(['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)
            
            df[['LotFrontage', 'MasVnrArea', 'garage_age']] = df[['LotFrontage', 'MasVnrArea', 'garage_age']].fillna(0)
            
            df.drop(['MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self, df):
        try:
            logging.info("Creating preprocessor with correct columns")
            
            numerical_features = df.select_dtypes(exclude="object").columns
            categorical_features = df.select_dtypes(include="object").columns

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_features),
                ('cat', cat_pipeline, categorical_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_preprocessing(self, train_data_path, test_data_path):
        try:
            logging.info("Reading training and testing data")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Separating features and target from train data")
            y_train = train_df['SalePrice']
            y_test = test_df['SalePrice'] if 'SalePrice' in test_df.columns else None
            X_train = train_df.drop(columns=['Id', 'SalePrice'], axis=1)
            X_test = test_df.drop(columns=['Id'], axis=1)

            X_train = self.feature_engineering(X_train)
            X_test = self.feature_engineering(X_test)

            preprocessor = self.get_preprocessor(X_train)

            logging.info("Fitting preprocessor on train data and transforming both train and test")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.preprocessing_config.preprocessor_path,
                obj=preprocessor
            )

            logging.info("Data preprocessing completed successfully")
            return (X_train_transformed, X_test_transformed,
                    y_train, y_test,
            self.preprocessing_config.preprocessor_path)

        except Exception as e:
            raise CustomException(e, sys)
