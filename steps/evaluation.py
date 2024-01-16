from sklearn.metrics import mean_squared_error, r2_score
import logging
from src.data_evaluation import Caluculation_configuration ,R2, MSE,RMSE
from typing import Annotated
def model_evaluation(model,
                     X_test,
                     y_test) -> tuple[
                         Annotated["float", "r2_score"],
                         Annotated["float", "rmse"]
                     ]:
    """model prediction process
    Args:
        model (obj): trained model object
        X_test (pd.DataFrame): X_test
        y_test (pd.DataFrame): y_test
    """
    
    prediction = model.predict(X_test)
    
    r2_class = R2()
    r2 = r2_class.caluculate_scores(y_test, prediction)

    mse_class = MSE()
    mse = mse_class.caluculate_scores(y_test, prediction)

    rmse_class = RMSE()
    rmse = rmse_class.caluculate_scores(y_test, prediction)

    logging.info("caluculating the model scores")

    return(
        r2,
        mse,
        rmse
    )