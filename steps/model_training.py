import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
def model_training_config(X_train,
                   X_test,
                   y_train,
                   y_test):
    """Train a linear regression model on the given training data and make predictions"""
    # Create an instance of the Linear Regression Model
    """model traning

    Raises:
        e: error occured in model training process

    Returns:
        trained model
    """
    try:
        model = LinearRegression()
        trained_model = model.fit(X_train, y_train)
        return trained_model

    except Exception as e:
        logging.error("error in above code {}".format(e))
        raise e