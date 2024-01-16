import pandas as pd
import logging
from typing import Dict, List
from steps.data_ingestion import process_post
from steps.clean_data import data_cleaning_configuration
from steps.model_training import model_training_config
from steps.evaluation import model_evaluation
from utils import data_cleaning_artifact, model_training_artifact

def train_data_pipeline(data_path: str):
    
    df = process_post(data_path=data_path)

    X_train, X_test, y_train, y_test = data_cleaning_configuration(df)
    trained_model = model_training_config(X_train, X_test, y_train, y_test)

    """saving the cleaned data model
    """
    model_training_artifact(trained_model)

    r2, mse, rmse = model_evaluation(trained_model,
                                   X_test,
                                   y_test)
    print(X_train)
    print(X_test.columns)
    
    

    
    
    