import pandas as pd
import numpy as np
import logging
from sklearn.metrics import r2_score, mean_squared_error
class Caluculation_configuration:
    def caluculate_scores(self, y_true: np.array, y_pred: np.array):
        try:
            """Calculate scores for model evaluation."""
            r2 = r2_score(y_true, y_pred)
            logging.info("r2 score {}".format(R2))
            return r2
        except Exception as e:
            logging.error("error in caluculate score {}".format(e))
            raise e

class R2:
    def caluculate_scores(self, y_true: np.array, y_pred: np.array):
        try:
            """Calculate scores for model evaluation."""
            r2 = r2_score(y_true, y_pred)
            logging.info("r2 score {}".format(R2))
            return r2
        except Exception as e:
            logging.error("error in caluculate score {}".format(e))
            raise e
        
class MSE:
    def caluculate_scores(self, y_true: np.array, y_pred: np.array):
        try:    
            logging.info("caluculating mean squared error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("mse score {}".format(mse))
            return mse
        except Exception as e:
            logging.error("error in MSE {}".format(e))
            raise e
        
class RMSE:
    def caluculate_scores(self, y_true: np.array, y_pred: np.array):
        try:    
            logging.info("caluculating root mean squared error")
            rmse = mean_squared_error(y_true, y_pred, squared= False)
            logging.info("rmse score {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("error in RMSE {}".format(e))
            raise e
        





