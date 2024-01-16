import pandas as pd
import numpy as np
from src.data_cleaning import Data_Cleaning, Data_Cleaning_Strategy, Data_Devide_Strategy
from utils import data_cleaning_artifact, model_training_artifact

def data_cleaning_configuration(df: pd.DataFrame) -> pd.DataFrame:
    data_clean = Data_Cleaning_Strategy()

    """data cleaning strategy artifct
    """
    data_cleaning_artifact(data_clean)

    data_clean_obj = Data_Cleaning(df, data_clean)
    clean_data = data_clean_obj.data_cleaing_process_strategy()
    data_devide = Data_Devide_Strategy()

    data_devide_obj = Data_Cleaning(clean_data, data_devide)
    X_train, X_test, y_train, y_test = data_devide_obj.data_cleaing_process_strategy()
    return X_train, X_test, y_train, y_test
    