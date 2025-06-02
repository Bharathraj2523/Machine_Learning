from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os 
import sys
import dill
from src.utils import load_object
import pandas as pd 

class PredictionPipeline:
    def __init__(self):
       pass

    def predict(seld , test_df):
        logging.info("test datafram from user reached predict function in pipeline")
        try:
            model_path=os.path.join("artifact","model.pkl")
            model = load_object(model_path)
            processor_path = os.path.join("artifact" , "preprocessor.pkl")
            preprocessor = load_object(processor_path)

            data_scaled = preprocessor.transform(test_df)
            preds = model.predict(test_df)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(
        self,
        temperature: float,
        irradiance: float,
        humidity: float,
        panel_age: float,
        maintenance_count: int,
        soiling_ratio: float,
        voltage: float,
        current: float,
        module_temperature: float,
        cloud_coverage: float,
        wind_speed: float,
        pressure: float,
        installation_type_encoded: int,
        error_code_E00: int,
        error_code_E01: int,
        error_code_E02: int,
        error_code_nan: int,
        area: float,
        power: float
    ):
        self.temperature = temperature
        self.irradiance = irradiance
        self.humidity = humidity
        self.panel_age = panel_age
        self.maintenance_count = maintenance_count
        self.soiling_ratio = soiling_ratio
        self.voltage = voltage
        self.current = current
        self.module_temperature = module_temperature
        self.cloud_coverage = cloud_coverage
        self.wind_speed = wind_speed
        self.pressure = pressure
        self.installation_type_encoded = installation_type_encoded
        self.error_code_E00 = error_code_E00
        self.error_code_E01 = error_code_E01
        self.error_code_E02 = error_code_E02
        self.error_code_nan = error_code_nan
        self.area = area
        self.power = power

    def get_data_as_data_frame(self):
        try:
            data = {
                "temperature": [self.temperature],
                "irradiance": [self.irradiance],
                "humidity": [self.humidity],
                "panel_age": [self.panel_age],
                "maintenance_count": [self.maintenance_count],
                "soiling_ratio": [self.soiling_ratio],
                "voltage": [self.voltage],
                "current": [self.current],
                "module_temperature": [self.module_temperature],
                "cloud_coverage": [self.cloud_coverage],
                "wind_speed": [self.wind_speed],
                "pressure": [self.pressure],
                "installation_type_encoded": [self.installation_type_encoded],
                "error_code_E00": [self.error_code_E00],
                "error_code_E01": [self.error_code_E01],
                "error_code_E02": [self.error_code_E02],
                "error_code_nan": [self.error_code_nan],
                "area": [self.area],
                "power": [self.power],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)

