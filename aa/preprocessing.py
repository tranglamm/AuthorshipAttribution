#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author Trang Lam - github.com/tranglamm

from pathlib import Path
import errno
import os
import sys 

#import gensim
from pickle import LIST

from numpy.lib.npyio import load
#import fasttext.util
#from fasttext import FastText
from .utils import *
from .preprocessing_utils import *
from .preprocessing_utils import *
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

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gzip 
import shutil
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder

import logging
logging.basicConfig(level=logging.DEBUG)


class PrepareData():
    def __init__(self,**kwargs):
        self.texts = kwargs.get("texts",None)
        self.labels = kwargs.get("labels",None)


    @classmethod
    def for_training(cls, params) -> "PrepareData":
        logging.info('Prepare data for Training...')
        logging.info(f'Loading {params.train_data} file...')
        texts,labels=read_csv_file(params.train_data,params.sep)
        

        #If labels is not integer
        if not all(isinstance(x, int) for x in labels):
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(np.array(labels))

        update_model_config({"num_classes":len(set(labels))})
        save_dict_labels(labels_encoded,labels)
        training_data = {"texts":texts,"labels":labels_encoded} 
        return cls(**training_data)
    
    @classmethod
    def for_prediction(cls,params) -> "PrepareData":
        """
        TODO: Predict can accept 
            only one sentence 
            csv_file and txt_file(done) 
        """
        logging.info('Prepare data for Prediction...')
        if params.txt_file:
            logging.info(f'Loading {params.txt_file} file...')
            with open(params.txt_file,"r",encoding="utf-8") as f: 
                texts=[l.strip('\n') for l in f.readlines()]
                predict_data={"texts":texts}
        elif params.csv_file:
            pass
        elif params.sentence: 
            pass 
        else: 
            raise Exception("No Data provided for prediction. Use one of three arguments : --txt_file --csv_file or --sentence")
        
        return cls(**predict_data)
        
    @staticmethod
    def transform_labels(y_train, y_val):
        labels= y_train + y_val
        labels_encoded = np.array(labels)
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        Y_train = label_encoder.transform(y_train)
        Y_val = label_encoder.transform(y_val)
        return Y_train, Y_val

    
    def get_train_data(self):
        train_data=[self.x_train,self.y_train]
        return train_data

    def get_val_data(self):
        val_data=[self.x_val,self.y_val]
        return val_data

    def test(self):
        predict_data=[self.x_test,self.y_test]
        return predict_data
    
class Preprocessing():
    def __init__(self, **kwargs):
        self.x_train = kwargs.get("x_train",None)
        self.y_train = kwargs.get("y_train",None)
        self.x_val = kwargs.get("x_val",None)
        self.y_val = kwargs.get("y_val",None)
        self.x_predict = kwargs.get("x_predict",None)
        self.y_predict = kwargs.get("y_predict",None)
        self.tokenizer = kwargs.get("tokenizer",None)
        

    @classmethod
    def for_training(cls,data) -> "Preprocessing":
        tokenizer= Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts(data.texts)
        logging.info(f'Split 10% training data for validation')
        x_train, x_val, y_train, y_val=split_data(data.texts,data.labels)
        x_train = cls.convert_data(tokenizer, x_train)
        x_val = cls.convert_data(tokenizer, x_val)
        y_train = cls.convert_label(data.labels,y_train)
        y_val = cls.convert_label(data.labels,y_val)
        training_data = {"x_train":x_train,"y_train":y_train,"x_val":x_val,"y_val":y_val,"tokenizer":tokenizer}
        return cls(**training_data)

    @classmethod
    def for_prediction(cls,data):
        data = data.texts
        word_index=load_word_index() 
        sentences=[]       
        for d in data: 
          sentence=[]
          words=d.split()
          for word in words: 
            if word in word_index: 
              sentence.append(word_index[word])
            else: 
              sentence.append(0)
          sentences.append((sentence))
        x_predict=np.array([np.array(sentence) for sentence in sentences])
        config=load_json('aa/ressources/config/model_config.json')
        x_predict = pad_sequences(x_predict, padding='post', maxlen=config['max_length'])
        predict_data={"x_predict":x_predict}
        return cls(**predict_data) 

    
    def get_vocab(self):
        """
        Word index: a dictionnary word-index based on word frequency
        Save word_index tokenizer to cache_dir 
        """
        save_word_index(self.tokenizer.word_index)
        return self.tokenizer.word_index

    def get_vocab_size(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        update_model_config({"vocab_size":vocab_size})
        return vocab_size
    
    @staticmethod
    def convert_data(tokenizer,x:List):
        """
        Convert to sequence of integers
        Transforms each text in data to a sequence of integers. 
        It takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary. 
        => Padding according to MAX_SEQUENCE_LENGTH
        """
        config=load_json('aa/ressources/config/model_config.json')
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, padding='post', maxlen=config['max_length'])
        return x 

    @staticmethod
    def convert_label(labels,y:List):
        y = to_categorical(y,num_classes=len(set(labels)))
        return y 

    def prepare_custom_embedding(self,emb_config):
        logging.info("Create Embedding Matrix")
        file_vec="aa/ressources/pretrained_emb/word2vec_%s.wordvectors" % emb_config.vector_size
        #file_vec="/content/drive/MyDrive/Colab Notebooks/AuthorshipAttribution/AuthorshipAttribution/aa/pretrained_emb/Campagne2017.vec"
        #wv=KeyedVectors.load_word2vec_format(file_vec, binary=False)
        logging.info(f"Create Embedding Matrix from {file_vec}")
        wv = KeyedVectors.load(file_vec, mmap='r')
        word_index=self.get_vocab()
        embedding_dim=emb_config.vector_size
        vocab_size = self.get_vocab_size() #
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in word_index.items():
          try:
            word_vec=wv[word]
          except:
            pass  
          else: 
            embedding_matrix[idx] = word_vec
        return embedding_matrix
        
    def prepare_txt_embedding(self,lang,emb_config):
        """
            Create emebdding matrix (from FastText embedding) for embedding layer of CNN model 
        """
        logging.info("Create Embedding Matrix")
        file_vec="aa/ressources/pretrained_emb/cc.%s.300.vec" % lang
        
        word_index=self.get_vocab()
        embedding_dim=emb_config.vector_size
        vocab_size = self.get_vocab_size() 
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
    
    @staticmethod
    def get_original_data(data):
        """
        Get original training data before having transformed to sequences of integers 
        """
        x_train, x_val, y_train, y_val=split_data(data.texts,data.labels)
        training_data=[x_train,y_train]
        validation_data = [x_val,y_val]
        return training_data,validation_data
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

    

