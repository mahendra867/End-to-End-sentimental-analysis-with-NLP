
import pandas as pd
import seaborn as sns
import os
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import operator
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Conv1D,LSTM,GRU,BatchNormalization,Flatten,Dense,Bidirectional,Dropout
import os
from NLP_PROJECT import logger
#from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle
from NLP_PROJECT.entity.config_entity import DataTransformationConfig




# here i defined the component of DataTransformationConfig below
class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def transform_the_target_feature(self):
        self.df = pd.read_csv(self.config.data_path)
        self.sentences=self.df['review']
        le=LabelEncoder()
        self.df['sentiment']= le.fit_transform(self.df['sentiment'])
        print(le.classes_)
        print(self.df['sentiment']) # according to label encoding transofrmation output 1 means positive , 0 means negative 
        #labels=to_categorical(self.df['sentiment'],num_classes=2)
        #print(labels)


        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.df['review'],self.df['sentiment'],test_size=0.1,random_state=10)
        print(f'This is x_train {self.X_train}')
        print(f'this is y_train {self.Y_train}')

        self.glove_embeddings= np.load("C:\\datascience End to End Projects\\capstone\\glove.840B.300d.pkl",
                          allow_pickle=True)
        logger.info("Done with word_cloud and loading the glove")

        return self.X_train,self.X_test,self.glove_embeddings
    

    def clean_sentences(self,line):
        
        line=re.sub('<.*?>','',line) # removing html tags
        
        #removing contractions
        line=re.sub("isn't",'is not',line)
        line=re.sub("he's",'he is',line)
        line=re.sub("wasn't",'was not',line)
        line=re.sub("there's",'there is',line)
        line=re.sub("couldn't",'could not',line)
        line=re.sub("won't",'will not',line)
        line=re.sub("they're",'they are',line)
        line=re.sub("she's",'she is',line)
        line=re.sub("There's",'there is',line)
        line=re.sub("wouldn't",'would not',line)
        line=re.sub("haven't",'have not',line)
        line=re.sub("That's",'That is',line)
        line=re.sub("you've",'you have',line)
        line=re.sub("He's",'He is',line)
        line=re.sub("what's",'what is',line)
        line=re.sub("weren't",'were not',line)
        line=re.sub("we're",'we are',line)
        line=re.sub("hasn't",'has not',line)
        line=re.sub("you'd",'you would',line)
        line=re.sub("shouldn't",'should not',line)
        line=re.sub("let's",'let us',line)
        line=re.sub("they've",'they have',line)
        line=re.sub("You'll",'You will',line)
        line=re.sub("i'm",'i am',line)
        line=re.sub("we've",'we have',line)
        line=re.sub("it's",'it is',line)
        line=re.sub("don't",'do not',line)
        line=re.sub("that´s",'that is',line)
        line=re.sub("I´m",'I am',line)
        line=re.sub("it’s",'it is',line)
        line=re.sub("she´s",'she is',line)
        line=re.sub("he’s'",'he is',line)
        line=re.sub('I’m','I am',line)
        line=re.sub('I’d','I did',line)
        line=re.sub("he’s'",'he is',line)
        line=re.sub('there’s','there is',line)
        
        #special characters and emojis
        line=re.sub('\x91The','The',line)
        line=re.sub('\x97','',line)
        line=re.sub('\x84The','The',line)
        line=re.sub('\uf0b7','',line)
        line=re.sub('¡¨','',line)
        line=re.sub('\x95','',line)
        line=re.sub('\x8ei\x9eek','',line)
        line=re.sub('\xad','',line)
        line=re.sub('\x84bubble','bubble',line)
        
        # remove concated words
        line=re.sub('trivialBoring','trivial Boring',line)
        line=re.sub('Justforkix','Just for kix',line)
        line=re.sub('Nightbeast','Night beast',line)
        line=re.sub('DEATHTRAP','Death Trap',line)
        line=re.sub('CitizenX','Citizen X',line)
        line=re.sub('10Rated','10 Rated',line)
        line=re.sub('_The','_ The',line)
        line=re.sub('1Sound','1 Sound',line)
        line=re.sub('blahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblahblah','blah blah',line)
        line=re.sub('ResidentHazard','Resident Hazard',line)
        line=re.sub('iameracing','i am racing',line)
        line=re.sub('BLACKSNAKE','Black Snake',line)
        line=re.sub('DEATHSTALKER','Death Stalker',line)
        line=re.sub('_is_','is',line)
        line=re.sub('10Fans','10 Fans',line)
        line=re.sub('Yellowcoat','Yellow coat',line)
        line=re.sub('Spiderbabe','Spider babe',line)
        line=re.sub('Frightworld','Fright world',line)
        
        #removing punctuations
        
        punctuations = '@#!~?+&*[]-%._-:/£();$=><|{}^' + '''"“´”'`'''
        for p in punctuations:
            line = line.replace(p, f' {p} ')
           
        line=re.sub(',',' , ',line)
           
        # ... and ..
        line = line.replace('...', ' ... ')
        
        if '...' not in line:
            line = line.replace('..', ' ... ')
        
            
        return line
        
    

    def vocab_build(self,review):
        logger.info("Done with cleaning data")
        logger.info("entered the vocab build")
    
        comments = review.apply(lambda s: s.split()).values
        self.vocab={}
        
        for comment in comments:
            for word in comment:
                try:
                    self.vocab[word]+=1
                    
                except KeyError:
                    self.vocab[word]=1
        
        logger.info("done with vocab build")

    def embedding_coverage(self,embeddings):
    
        #self.vocab=self.vocab_build(review)
        
        covered={}
        word_count={}
        oov={}
        covered_num=0
        oov_num=0
        
        for word in self.vocab:
            try:
                covered[word]=embeddings[word]
                covered_num+=self.vocab[word]
                word_count[word]=self.vocab[word]
            except:
                oov[word]=self.vocab[word]
                oov_num+=oov[word]
        
        vocab_coverage=len(covered)/len(self.vocab)*100
        text_coverage = covered_num/(covered_num+oov_num)*100
        
        sorted_oov=sorted(oov.items(), key=operator.itemgetter(1))[::-1]
        sorted_word_count=sorted(word_count.items(), key=operator.itemgetter(1))[::-1]
        
        return sorted_word_count,sorted_oov,vocab_coverage,text_coverage
        





    def embedding_coverage_test(self,train_covered, train_oov, train_vocab_coverage, train_text_coverage):
        self.train_test_covered = train_covered
        self.train_test_oov = train_oov
        self.train_test_vocab_coverage = train_vocab_coverage
        self.train_test_text_coverage = train_text_coverage

        print(f"Glove embeddings cover {round(self.train_test_vocab_coverage,2)}% of vocabulary and {round(self.train_test_text_coverage,2)}% text in training set")

        return self.train_test_covered,self.train_test_oov


    def train_word_and_train_word_count(self,train_covered):
        self.punctuations = '@#!~?+&*[]-%._-:/£();$=><|{},^' + '''"“´”'`'''
        train_word=[]
        train_count=[]

        i=1
        for word,count in train_covered: 
            if word not in self.punctuations:
                train_word.append(word)
                train_count.append(count)
                i+=1
            if(i==15):
                break

    def test_word_and_test_word_count(self,test_covered):
        test_word=[]
        test_count=[]

        i=1
        for word,count in test_covered: 
            if word not in self.punctuations:
                test_word.append(word)
                test_count.append(count)
                i+=1
            if(i==15):
                break

    def deleting_out_of_vocab(self,train_oov,test_oov):
        if self.glove_embeddings is not None:
            del self.glove_embeddings
        if hasattr(self, 'train_oov'):  # Check if train_oov is defined
            del train_oov
        if hasattr(self, 'test_oov'):  # Check if test_oov is defined
            del test_oov
        gc.collect()


    def find_vocab_word_count(self,X_train_cln):
        self.num_words=80000
        self.embeddings=256
        self.tokenizer=Tokenizer(num_words=self.num_words,oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train_cln)
        word_index=self.tokenizer.word_index
        total_vocab=len(word_index)
        print("Vocabulary of the dataset is : ",total_vocab)


    def creating_train_and_test_pad(self):
        
        sequences_train=self.tokenizer.texts_to_sequences(self.X_train)
        sequences_test=self.tokenizer.texts_to_sequences(self.X_test)
        
        with open(os.path.join(self.config.root_dir, self.config.preprocessor_obj), 'wb') as tokenizer_pkl_file:
            pickle.dump(self.tokenizer, tokenizer_pkl_file)

        max_len=max(max([len(x) for x in sequences_train]),max([len(x) for x in sequences_test]))
        print(f'the max_len is {max_len}')
        train_padded = pad_sequences(sequences_train, maxlen=max_len)
        test_padded = pad_sequences(sequences_test, maxlen=max_len)
    
        X_train,X_val,Y_train,Y_val=train_test_split(train_padded,self.Y_train,
                                             test_size=0.05,random_state=10)

        model = keras.Sequential()
        model.add(Embedding(self.num_words, self.embeddings, input_shape=(max_len,)))
        model.add(Conv1D(256, 10, activation='relu'))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(LSTM(64))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()


        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
             )

        # here i need to add early stopping if need based on my model performance

        logger.info("initiated model training")

        history=model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=5,batch_size=32)

        logger.info("Done with model Training")
        model.save(os.path.join(self.config.root_dir, self.config.model_file))


        return history

    def plot_graph(history,string):
    
        plt.plot(history.history[string],label='training '+string)
        plt.plot(history.history['val_'+string],label='validation '+string)
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel(string)
        plt.title(string+' vs epochs')
        plt.show()

