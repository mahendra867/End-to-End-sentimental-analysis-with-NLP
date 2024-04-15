import os
import zipfile # here iam importing the zipfile to unzip the chest images data
import gdown  # here iam importing the gdown to download the chest image data from google drive
from NLP_PROJECT import logger 
from NLP_PROJECT.utils.common import get_size  # here iam importing the get_size to get to know the size of the data
from NLP_PROJECT.entity.config_entity import DataIngestionConfig 


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try:  # do follow the explaination of trais.ipynb file for to know how to download the data from googledrive
            dataset_url = self.config.source_URL  # here iam getting the dataset url 
            zip_download_dir = self.config.local_data_file  # here iam saving the path of the images data of google drive 
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]  # 
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):  # here by using the extract_zip_file fucntion we are  going to unzip the images dataset 
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)