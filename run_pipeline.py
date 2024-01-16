import pandas as pd
from pipelines.train_pipeline import train_data_pipeline
import logging

data_path = r"D:\my projects\customer-satisfaction-prediction\data\raw\olist_customers_dataset.csv"
logging.info("data path passes to the train_pipeline")
train_data_pipeline(data_path)

