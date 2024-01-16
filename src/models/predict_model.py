import pandas as pd
import numpy as np
import pickle


def trained_model_read(file, df):
    with open(file, 'rb') as f:
            pcl = pickle.load(f)
            trained_model = pcl.predict(df)
            return trained_model