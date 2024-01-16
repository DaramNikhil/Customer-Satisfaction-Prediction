import logging
import numpy as np

class Model_Configuration:
    try:

        configure:str = "LinearRegression"
        
    except Exception as e:
        logging.error("error in model cofniguration")
        raise e