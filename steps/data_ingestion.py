import pandas as pd
import numpy as np
import logging

class Data_Ingestion:
    
    def __init__(self, data_path: str):
        """

        Args:
            data_path (str): passes the data_path
        """
        self.data_path = data_path

    def Data_Ingestion_Configuration(self)->pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: It will returns Dataframe
        """
        return pd.read_csv(self.data_path)
    

def process_post(data_path)->str:
    Ingestion = Data_Ingestion(data_path)
    df=Ingestion.Data_Ingestion_Configuration()
    logging.info("data ingestion return data")
    return df
    