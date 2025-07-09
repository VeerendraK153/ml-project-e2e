import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.components.data_preprocessing import DataPreprocessing

class PredictPipeline:
    def __init__(self):
        # Load preprocessor and model
        self.preprocessor = load_object('artifacts/preprocessor.pkl')
        self.model = load_object('artifacts/model.pkl')  # Adjust if your model file is named differently

    def predict(self, features):
        try:
            # Convert input dictionary to DataFrame
            df = pd.DataFrame([features])

            # Create DataPreprocessing instance
            data_preprocessor = DataPreprocessing()

            # Apply feature engineering
            df_fe = data_preprocessor.feature_engineering(df)

            # Transform using preprocessor
            processed = self.preprocessor.transform(df_fe)

            # Predict using loaded model
            prediction = self.model.predict(processed)

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)
