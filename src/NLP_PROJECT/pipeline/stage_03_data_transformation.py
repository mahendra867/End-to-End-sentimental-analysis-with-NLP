from NLP_PROJECT.config.configuration import ConfigurationManager
from NLP_PROJECT.components.data_transformation import DataTransformation
from NLP_PROJECT import logger
from pathlib import Path




STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            x_train,x_test,glove_embeddings=data_transformation.transform_the_target_feature()
            X_train_cleaned = data_transformation.X_train.apply(lambda s: data_transformation.clean_sentences(s))  # Corrected usage without 'self'
            X_test_cleaned = data_transformation.X_test.apply(lambda s: data_transformation.clean_sentences(s))  # Corrected usage without 'self'
            data_transformation.vocab_build(X_train_cleaned)  # Pass the required parameter here
            train_covered, train_oov, train_vocab_coverage, train_text_coverage = data_transformation.embedding_coverage(glove_embeddings)  # Pass the required parameters here
            train_covered_for_word_count,train_data_oov=data_transformation.embedding_coverage_test(train_covered, train_oov, train_vocab_coverage, train_text_coverage)
            data_transformation.vocab_build(X_test_cleaned)
            test_covered, test_oov, test_vocab_coverage, test_text_coverage = data_transformation.embedding_coverage(glove_embeddings)  # Pass the required parameters here
            test_covered_for_word_count,test_data_oov=data_transformation.embedding_coverage_test(test_covered, test_oov, test_vocab_coverage, test_text_coverage)    
            data_transformation.train_word_and_train_word_count(train_covered_for_word_count)
            data_transformation.test_word_and_test_word_count(test_covered_for_word_count)
            data_transformation.deleting_out_of_vocab(train_data_oov,test_data_oov)
            data_transformation.find_vocab_word_count(X_train_cleaned)
            history=data_transformation.creating_train_and_test_pad()
            data_transformation.plot_graph(history,'loss')
            data_transformation.plot_graph(history,'accuracy')

        except Exception as e:
            print(e)





if __name__ == '__main__': # here i have initlized the main fucntion 
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e