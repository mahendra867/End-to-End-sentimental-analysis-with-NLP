
from NLP_PROJECT.constants import * # here iam importing everthing which is present in the constants->__init__.py file into inside the data_ingestion.ipynb
from NLP_PROJECT.utils.common import read_yaml, create_directories # here iam importing the read_yaml, create_directories which are presenting inside the utils,common files into PROJECTML in which the file is data_ingestion.ipynb
from NLP_PROJECT.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig


class ConfigurationManager:  # here iam creating class called ConfigurationManager
    def __init__( # inisde this class iam reading all the yaml files which iam calling it from constants->__init__.py file and iam mentioning inside the class varaiable 
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath) # and here iam giving read_yaml path here and iam giving the path after that then it will return all the configuration in the variable
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

# now i will create artifacts root in the side of the vscode project one of the path and the below i will define the data ingestion cofiguration function
    # the above one  entity which inside 4 variables needs to return by this below fucntion
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir]) # here iam creating the root directory, and iam reading the config from the configurationManager class and iam going to access all the data ingestion from the config.yaml file 

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,  # that how iam accessing all the things like root_dir,source_url and etc from config.yaml file and finally this fucntion do return all this variables 
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    # then i will prepare my get_data_validation_config , so inside this iam returning all of the varaibles that i have deifined inside my entity 
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation  # after reading by config iam returning the root_dir,status_file etc one by one
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig( # the above entity code is return type , and the below varaibles are getting return after reading by config varaible 
            root_dir=config.root_dir, 
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

        # only this part get changes in every step, only defining the get_data_transformation_config get changes according to which step we are performing like 01_data_ingestion,02_data_validation
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,  # here iam returning these 2 varaibles by using this code 
            data_path=config.data_path,
            preprocessor_obj=config.preprocessor_obj,
            model_file=config.model_file,
        )

        return data_transformation_config