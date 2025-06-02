import os
import sys
import pandas as pd 

from dataclasses import dataclass
from src.exception import CustomException
from src.logger    import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer       import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_file_path = os.path.join("artifact","train.csv")
    raw_file_path   = os.path.join("artifact" , "raw.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config_obj = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Created a Data Frame train_df")
            raw_df = pd.read_csv('notebook\\data\\train.csv')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config_obj.raw_file_path), exist_ok=True)

            logging.info("Created a Train.csv file")
            raw_df.to_csv(self.data_ingestion_config_obj.raw_file_path , index= False , header= True)

            
            return self.data_ingestion_config_obj.raw_file_path
        
        except Exception as e:
            logging.info("Error Ocurred in Data Ingestion File ")
            raise CustomException(e , sys)
        

if __name__ == "__main__":
    obj  = DataIngestion()
    raw_path = obj.initiate_data_ingestion()
    print(raw_path)

    trans_obj = DataTransformation()
    x_path , y_path = trans_obj.initiate_data_transformation(raw_path)

    print(x_path)

    train_obj = ModelTrainer()
    print(f"R2 Score is {train_obj.initiate_model_trainer(x_path,y_path)}")



