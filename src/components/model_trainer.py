import pandas as pd 
import os 
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model , save_object

import xgboost as xgb

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifact" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self ,train , test):

        '''
        in this function we are gng to train a model and store it in pkl file

        '''
        logging.info("initiate model trainer function is called ")
        train_df = pd.read_csv(train)
        test_df  = pd.read_csv(test)

        input_col =     [ "temperature", "irradiance", "humidity", "panel_age", "maintenance_count", "soiling_ratio", "voltage", "current", "module_temperature",
                        "cloud_coverage", "wind_speed", "pressure","installation_type_encoded", "error_code_E00", "error_code_E01", "error_code_E02", "error_code_nan",
                        "area","power" ]
        target_col =    ['efficiency']

        logging.info("splitting the train test df for model training and validating ")
        x_train, y_train = train_df[input_col], train_df[target_col]
        x_test, y_test = test_df[input_col], test_df[target_col]

        logging.info("initializing the model ")
        model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=12,
        subsample=0.9,
        colsample_bytree=0.8,
        n_jobs=-1
        )

        logging.info("Calling the evaluate model ")
        predicted_score = evaluate_model( x_train , y_train , x_test , y_test , model)
        logging.info(f"Predicted score is {predicted_score}")
        save_object(
            self.config.model_path , 
            model,
        )

        return predicted_score


        