import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from typing import Union

class Data_Configuration:
    def __init__(self) -> None:
        pass



class Data_Cleaning_Strategy:
    """data cleaning strategy implimentation
    """
    
    def handle_data(self, data: pd.DataFrame) ->pd.DataFrame:
        """In this class we can preprocess the given data

        Args:
            data (pd.DataFrame): data geting from the above source

        Returns:
            pd.Dataframe
        """
        data = data.drop([
                "order_approved_at",
            ],
              axis=1)
            
        data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
        data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
        data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
        data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

        data = data.select_dtypes(include=[np.number])

        cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
        data = data.drop(cols_to_drop, axis=1)
        return data
    

class Data_Devide_Strategy:
    
    def handle_data(self, data: pd.DataFrame):
        """

        Args:
            data (pd.DataFrame): data geting from the above source

        Returns:
            X_train,
            X_test,
            y_train,
            y_test
        """
        X = data.drop(["review_score"], axis=1)
        y = data["review_score"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        return X_train, X_test,y_train, y_test
    
class Data_Cleaning:
    def __init__(self, data, strategy) -> Union[pd.DataFrame, np.array]:
        
        self.data = data
        self.strategy = strategy

    def data_cleaing_process_strategy(self):
        """

        Raises:
            e: error in data cleaning process

        Returns:
            cleaned_data,
            X_train,
            X_test,
            y_train,
            y_test
        """
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            logging.error("error in data cleaning process strategy")
            raise e