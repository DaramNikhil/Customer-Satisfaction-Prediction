import pandas as pd
import numpy as np
import logging
import os
import pickle

def data_cleaning_artifact(data_clean_obj: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        data_clean_obj (train data obj): store the train data object

    Returns:
        pd.DataFrame: cleaned data
    """
    
    data_path = os.path.join("models","preprocess.pkl")

    os.makedirs(os.path.dirname(data_path),exist_ok=True)

    with open(data_path, "wb") as f:
        pickle.dump(data_clean_obj, f)



def model_training_artifact(model):
    """model saving

    Args:
        model (model obj): machine learning algorithms
    """
    data_path = os.path.join("models","model.pkl")
    
    os.makedirs(os.path.dirname(data_path),exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(model, f)