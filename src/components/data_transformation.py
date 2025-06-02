from dataclasses import dataclass
import numpy as np 
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import predict_col , fill_voltage_current_power_irradiance ,save_object

@dataclass
class DataTransformationConfig:
    train_file_path = os.path.join("artifact","train.csv")
    test_file_path  = os.path.join("artifact" , "test.csv")
    cleaned_file_path = os.path.join("artifact" , "cleaned.csv")
    preprocessor_file_path = os.path.join("artifact" ,"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def fill_null_values(self , rawdf):
        '''
        In This function we predict the missing features Using XGBoost Regressor 

        '''
        logging.info("Fill Null values function called")

        # Columns
        numerical_cols = ['temperature' , 'irradiance', 'humidity', 'panel_age' , 'maintenance_count'  ,  'soiling_ratio', 'voltage', 'current',
            'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure' , 'efficiency']
        categorical_cols = ['error_code']
        # ['installation_type'] this column Label Encoded
        # Some of these Numerical columns are of type Object so Make them numeric 
        logging.info("Converting object type col to numeric")

        for col in numerical_cols:
            rawdf[col] = pd.to_numeric(rawdf[col], errors='coerce')

        # Now encode the categorical col 
        logging.info("Encoding the Error Col")
        encoder = OneHotEncoder().fit(rawdf[["error_code"]])
        encoder_list = list(encoder.get_feature_names_out())
        rawdf[encoder_list] = encoder.transform(rawdf[["error_code"]]).toarray()

        # Now Lets label the installation_type col
        logging.info("Label Encoding the ['installation_type']")
        known_df = rawdf[rawdf['installation_type'].notna()].copy()
        le = LabelEncoder()
        known_df['installation_type_encoded'] = le.fit_transform(known_df['installation_type'])
        rawdf['installation_type_encoded'] = np.nan
        rawdf.loc[known_df.index, 'installation_type_encoded'] = known_df['installation_type_encoded']

        # Now lets create col ['power] , ['area']
        rawdf['power'] = rawdf['voltage'] * rawdf['current']
        rawdf['area'] = rawdf["power"] / (rawdf['irradiance'] * rawdf["efficiency"] )

        # some of them are type object so lets numerise them 
        cat_col = [ "installation_type_encoded",'error_code_E00','error_code_E01', 'error_code_E02', 'error_code_nan']
        for col in numerical_cols + cat_col:
             rawdf[col] = pd.to_numeric(rawdf[col], errors='coerce')
        rawdf['area'] = rawdf['area'].replace([np.inf, -np.inf], 0)

        #Lets split the input split for training the model
        rawdf['area'] = rawdf['area'].replace([np.inf, -np.inf], 0)
        rawdf['power'] = rawdf['power'].replace([np.inf, -np.inf], 0)
        input_df = rawdf[numerical_cols + cat_col + ['area' , 'power']]

        # Lets seperate the col that are null and those are targeted columns 
        missing_cols = []
        for col in numerical_cols + cat_col:
            if rawdf[col].isna().sum() != 0:
                missing_cols.append(col)
        missing_cols += ['area', 'power']  # append once, not inside the loop
        logging.info(f"missing values in cols {missing_cols}")

        #Before training remove the target feature from input 
        rawdf['area'] = rawdf["power"] / (rawdf['irradiance'])

        # Lets train the model 
        for col in missing_cols:
            rawdf = predict_col(input_df, rawdf, numerical_cols, cat_col , col)


        return rawdf
    
    def fill_null_statistically(self , new_df):
        '''
        in this function rows that cant be predicted are statiscally predicted using mean median and etc

        '''
        logging.info("fill_null_statiscally function is called")

        # fill_null_statiscally function is called
        logging.info("Randomly filling installation_type with 0 1 2 ")
        new_df['installation_type_encoded'] = pd.cut(
        new_df['installation_type_encoded'],
        bins=[-float('inf'), 0.5, 1.5, float('inf')],
        labels=[0, 1, 2]
        )
        new_df['installation_type_encoded'] = pd.to_numeric(new_df['installation_type_encoded'], errors='coerce')
        missing_mask = new_df['installation_type_encoded'].isna()
        new_df.loc[missing_mask, 'installation_type_encoded'] = np.random.choice([0, 1, 2], size=missing_mask.sum())

        # Define average area mapping
        avg_area_map = { 0: 0.061317, 1: 0.066419, 2: 0.061367 }
        
        logging.info("Fill area with avg area grouped by installation_type")
        # Fill missing area directly using map
        area_missing_mask = new_df['area'].isna()
        new_df.loc[area_missing_mask, 'area'] = new_df.loc[area_missing_mask, 'installation_type_encoded'].astype(int).map(avg_area_map)

        new_df = fill_voltage_current_power_irradiance(new_df)

        # from the dataset we conclude that temperature is directly proportional to module temperature
        # lets fill temperature values nan with moduletemperature
        # Vice versa for moduletemperature
        logging.info("filling temperature and modular temperature ")
        new_df["temperature"] = new_df["temperature"].fillna(new_df["module_temperature"])
        new_df["module_temperature"] = new_df["module_temperature"].fillna(new_df["temperature"])

        # Lets fill them with median
        logging.info("filling rest of the col")
        new_df["soiling_ratio"] = new_df["soiling_ratio"].fillna(new_df["soiling_ratio"].median())
        new_df["wind_speed"] = new_df["wind_speed"].fillna(new_df["wind_speed"].median())
        new_df["cloud_coverage"] = new_df["cloud_coverage"].fillna(new_df["cloud_coverage"].median())
        new_df["pressure"] = new_df["pressure"].fillna(new_df["pressure"].median())

        # Replace infinite values with NaN so we can handle them uniformly
        new_df['irradiance'].replace([np.inf, -np.inf], np.nan, inplace=True)
        new_df['current'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Now fill those NaNs — you can use median or mean depending on distribution
        new_df['irradiance'].fillna(new_df['irradiance'].median(), inplace=True)
        new_df['current'].fillna(new_df['current'].median(), inplace=True)

        return new_df
    
    def get_data_transformer_obj(self,input_col):
        pipeline = Pipeline(
            steps=[
                ("scale" ,StandardScaler())
            ]
        )
        
        preprocessor = ColumnTransformer(
            [
                ("scaler" , pipeline , input_col)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self , raw_data_path):

        self.data_trans_obj = DataTransformationConfig()

        logging.info("initiate data ingestion called")
        df = pd.read_csv(raw_data_path)
        logging.info(f"Read raw data with shape: {df.shape}")
        logging.info(f"Missing values before prediction:\n{df.isnull().sum()}")

        
        df_filled = self.fill_null_values(df)
        # Now that we have filled almost 70% but still there are some so we have to statistically fill them 
        df_filled = self.fill_null_statistically(df_filled)
        logging.info("after filling most of the values lets drop row with Nan")        
        df_filled = df_filled.dropna()

        # Now we have dataset that is filled almost and ready for test train split but before that lets scale it 
        input_col =     [ "temperature", "irradiance", "humidity", "panel_age", "maintenance_count", "soiling_ratio", "voltage", "current", "module_temperature",
                        "cloud_coverage", "wind_speed", "pressure","installation_type_encoded", "error_code_E00", "error_code_E01", "error_code_E02", "error_code_nan",
                        "area","power" ]
        target_col =    ['efficiency']
        
        preprocessor = self.get_data_transformer_obj(input_col)

        df_filled.to_csv(self.data_transformation_config.cleaned_file_path , index= False , header= True)

        logging.info("splitting train and test set")
        train_df , test_df = train_test_split(df_filled , test_size= 0.2 , random_state=42)
        input_feature_train_df = train_df.drop( columns=target_col , axis=1)
        target_feature_train_df = train_df[target_col]

        input_feature_test_df = test_df.drop( columns=target_col , axis=1)
        target_feature_test_df = test_df[target_col]

        logging.info("Preprocess the spits")        
        input_feature_train_arr =  preprocessor.fit_transform(input_feature_train_df)
        input_feature_test_arr =  preprocessor.transform(input_feature_test_df)

        train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
        test_arr  = np.c_[input_feature_test_arr,  target_feature_test_df]

        # Convert to DataFrame with columns
        train_df_final = pd.DataFrame(train_arr, columns = input_col + target_col)
        test_df_final  = pd.DataFrame(test_arr,  columns = input_col + target_col)

        # Save
        train_df_final.to_csv(self.data_transformation_config.train_file_path, index=False)
        test_df_final.to_csv(self.data_transformation_config.test_file_path, index=False)

        #save preprocessor
        save_object(
            self.data_trans_obj.preprocessor_file_path,
            preprocessor
        )



        return (
            self.data_transformation_config.train_file_path,
            self.data_transformation_config.test_file_path

        )



        

