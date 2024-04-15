import h5
import numpy as np
import pandas as pd
from pathlib import Path
from NLP_PROJECT import logger
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from NLP_PROJECT.utils.common import load_object1



class CustomData:
        def __init__(
            self,
            review: object,
            #sentiment: object,
        ):

            self.review = review
            
            logger.info("feature values got stored inside the CustomData class")

        def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = { # here i passing  the features (which has values which passed by user) in sequence of how the original dataset has features sequence
                    "review": [self.review]
                    
                }
                self.df = pd.DataFrame(custom_data_input_dict)
                logger.info("Dataframe Gathered")
                logger.info(f"Dataframe gathered values are {self.df}")
                return self.df
                

            except Exception as e:
                logger.info("Exception Occurred in prediction pipeline")
                raise e


class PredictionPipeline:
        def __init__(self):
            #self.model = load_bin("artifacts/model_trainer/model.pkl")
            self.model=load_model('C:/datascience End to End Projects/End-to-End-sentimental-analysis-with-NLP/new_imdb_model.h5')
            logger.info(f"loaded model object preview is : {self.model}")

            self.tokenizer = load_object1('artifacts/data_transformation/tokenizer_pkl_file')
            logger.info(f"loaded preprocessor tokenizer object preview is : {self.tokenizer}")
            

            #self.tokenizer = pickle.load('artifacts/data_transformation/tokenizer_pkl_file')

            print(self.model)
            logger.info("Model+preprocessing objects loaded successfully")
    
    
        def predict(self,data):
            logger.info("initiated converting the user data to tokens")
            data=self.tokenizer.texts_to_sequences(data)
            logger.info(f"tokenized converted Input data: {data}")
            logger.info("done with conversion")
            max_len = 2527  # Or use the max_len used during training
            data = pad_sequences(data, maxlen=max_len)
            logger.info(f"padded sequence Input data: {data}")
            # write the code here what  i need to perform to predict the review sample user send data
            prediction = self.model.predict(data)
            logger.info("Model predicted the Data")
            logger.info(f"Input data: {data}")
            logger.info(f"Predicted output: {prediction}")
            return prediction
