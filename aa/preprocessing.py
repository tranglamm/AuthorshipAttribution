#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
import errno
import os
import sys 

#import gensim
from pickle import LIST

from numpy.lib.npyio import load
#import fasttext.util
#from fasttext import FastText
from .preprocessing_utils import *
from .preprocessing_utils import clean_tweet, load_json, read_csv_file, save_dict_labels, split_data, transform_labels, inverse_labels, load_dict_labels, save_dict_labels
#from .config_utils import TrainConfig, EmbConfig
import numpy as np
import io
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
#Tokenizer: a sentence with a list of string (tokens), split of string 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import gzip 
import shutil
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder

import logging
logging.basicConfig(level=logging.DEBUG)
"""
def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)
"""


class PrepareData():
    def __init__(self,params):
        #If Train: 
        if sys.argv[0] == "train.py":
            logging.info('Prepare data for Training...')
            if params.train_data and params.val_data:
                logging.info(f'Loading {params.train_data} and {params.val_data} file...')
                x_train, y_train = read_csv_file(params.train_data)
                x_val, y_val = read_csv_file(params.val_data)
                texts=x_train+x_val
                labels=y_train+y_val
            elif params.train_data: 
                logging.info(f'Loading {params.train_data} file...')
                texts,labels=read_csv_file(params.train_data)
                x_train, x_val, y_train, y_val=split_data(texts,labels)
            else: 
                print("You have to pass as least --train_data argument")

            #If labels is not integer
            if not all(isinstance(x, int) for x in labels):
                y_train, y_val = transform_labels(y_train,y_val)
            labels_encoded = np.concatenate((y_train,y_val))

            save_dict_labels(labels,labels_encoded)
            self.dict_labels = inverse_labels(labels_encoded,labels)
            self.texts = texts
            self.labels = labels
            update_model_config({"num_classes":len(set(labels))})
            self.train_data = [x_train, y_train]
            self.val_data = [x_val, y_val]  

            # If Test: Review : Here we can pass a list of sentences or a csv
            """
            elif params.test: 
                
                Review : Here we can pass a list of sentences or a csv
                texts,labels = read_csv_file(params.test_data)
                dict_labels=load_dict_labels()
                if not all(isinstance(x, int) for x in labels):
                    tmp_df=pd.DataFrame(labels)
                    y_test= [label.get(k) for label in labels if label find ]
                self.test_data 
            """

        #Predict only one sentence 
        elif sys.argv[0] == "predict.py":  
            if params.sentence and params.true_target: 
                self.sentence=params.sentence
                true_target=params.true_target
                dict_labels=load_dict_labels()
                if true_target in dict_labels: 
                    self.true_target=[dict_labels.get(true_target)]
                    self.predict_data=[self.sentence,self.true_target]
                else: 
                    print("It seems that the TARGET of your data can not be found in your trained model ... ")
            else: 
                print("2 arguments ... ") 

           
            
    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_predict_data(self):
        return self.predict_data

    def get_dict_labels(self):
        """
        Dict of labels: {0: label1, 1: label2, ...}
        """
        return self.dict_labels
    
    def get_texts(self):
        return self.texts
        #return self.train_data[0]+self.val_data[0]
    
    def get_labels(self):
        return self.labels
        #return self.train_data[1]+self.val_data[1]

class Preprocessing():
    def __init__(self, params, 
                #**kwargs
                ):
        #super().int(**kwargs)
        #Prepare data : get data 
        
        self.data = PrepareData (params)
        self.texts = self.data.get_texts()
        self.params=params    
        #self.tokenizer=Tokenizer(**kwargs)
        self.tokenizer= Tokenizer()
        self.tokenizer.fit_on_texts(self.data.get_texts())

    def get_training_data(self):
        train_data = self.data.get_train_data()
        x_train = self.convert_data(train_data[0])
        y_train = self.convert_label(train_data[1])
        return x_train, y_train 

    def get_validation_data(self):
        val_data = self.data.get_val_data()
        x_val = self.convert_data(val_data[0])
        y_val = self.convert_label(val_data[1])
        return x_val, y_val
    
    def get_predict_data(self):
        predict_data = self.data.get_predict_data()
        x_predict = self.convert_data(predict_data[0])
        y_predict = self.convert_data(predict_data[1])

    def get_vocab(self):
        """
        Word index: a dictionnary word-index based on word frequency
        Save word_index tokenizer to cache_dir 
        """
        return self.tokenizer.word_index

    def get_vocab_size(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        update_model_config({"vocab_size":vocab_size})
        return vocab_size

    def convert_data(self,x:List):
        """
        Convert to sequence of integers
        Transforms each text in data to a sequence of integers. 
        It takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. 
        => Padding according to MAX_SEQUENCE_LENGTH
        """
        x = self.tokenizer.texts_to_sequences(x)
        config=load_json('aa/config/model_config.json')
        x = pad_sequences(x, padding='post', maxlen=config['max_length'])
        return x 
    
    def convert_label(self,y:List):
        labels=self.data.get_labels()
        y = to_categorical(y,num_classes=len(set(labels)))
        return y 

    def prepare_custom_embedding(self,emb_config):
        logging.info("Create Embedding Matrix")
        """
        file_vec="aa/pretrained_emb/word2vec.%s.wordvectors" % emb_config.vector_size
        logging.info(f"Create Embedding Matrix from {file_vec}")
        wv = KeyedVectors.load(file_vec, mmap='r')
        word_index=self.get_vocab()
        #embedding_dim=self.config.emb_dim
        embedding_dim=emb_config.vector_size
        vocab_size = self.get_vocab_size() # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in word_index.items():
            if wv[word] is not None:
                embedding_matrix[idx] = wv[word]
            '''
            #read first line
            n, d = map(int, file.readline().split())
            for line in file:
                values=line.rstrip().split(' ')
                word=values[0]
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
            '''
        return embedding_matrix
        """
    def prepare_txt_embedding(self,emb_config):
        logging.info("Create Embedding Matrix")
        #lang=self.params.lg
        lang="fr"
        file_vec="aa/pretrained_emb/cc.%s.300.vec" % lang
        
        word_index=self.get_vocab()
        #embedding_dim=self.config.emb_dim
        embedding_dim=emb_config.vector_size
        vocab_size = self.get_vocab_size() # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        with io.open(file_vec, 'r', encoding="utf-8",newline='\n', errors='ignore') as file:
            #read first line
            n, d = map(int, file.readline().split())
            for line in file:
                values=line.rstrip().split(' ')
                word=values[0]
                if word in word_index:
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
        return embedding_matrix
    """
    def prepare_bin_embedding(self,file_name):
        #Get vocab= word_index
        word_index=self.get_vocab()
        #Get dimension of FastText embedding 
        embedding_dim=300
        #Get size of vocab == word_index
        vocab_size=self.get_vocab_size()
        #Initialize embedding_matrix 
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word,idx in word_index.items():
            try: 
                embedding_matrix[idx] = ft.get_word_vector(word)
            except: 
                pass
        return embedding_matrix
    """
    
    
"""
if __name__=="__main__":
    preprocessing=Preprocessing()
    embedding_matrix=
"""

    

