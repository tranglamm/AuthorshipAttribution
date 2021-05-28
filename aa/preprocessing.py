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
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
#import fasttext.util
#from fasttext import FastText
from .utils import *
from .preprocessing_utils import *
from .preprocessing_utils import *
#from .config_utils import TrainConfig, EmbConfig
import numpy as np
import io
import pandas as pd
from .file_utils import CONFIG_JSON
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
#Tokenizer: a sentence with a list of string (tokens), split of string 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import torch 

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gzip 
import shutil
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
from torch.utils.data import Dataset, DataLoader



class PrepareData():
    def __init__(self,**kwargs):
        self.texts = kwargs.get("texts",None)
        self.labels = kwargs.get("labels",None)
        self.max_length = kwargs.get("max_length",None)

    @classmethod
    def for_transformer(cls,params) -> "PrepareData":
        model_name="/".join(params.model.split("/")[1:])
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings=params.max_length if params.max_length and params.max_length <= config.max_position_embeddings else config.max_position_embeddings
        
        texts,labels=read_csv_file(params.train_data,params.sep)
        data = {"texts":texts,"labels":labels,"max_length":config.max_position_embeddings}
        
        #If labels is not integer
        if not all(isinstance(x, int) for x in labels):
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(np.array(labels))

        config.num_labels=len(set(labels))
        config.save_pretrained("aa/ressources/config")
        save_dict_labels(labels_encoded,labels)
        return cls(**data)

    @classmethod
    def for_training(cls, params) -> "PrepareData":
        logging.info('Prepare data for Training...')
        logging.info(f'Loading {params.train_data} file...')
        max_length=params.max_length if params.max_length else 50
        update_model_config({"max_length":max_length})
        texts,labels=read_csv_file(params.train_data,params.sep)
        

        #If labels is not integer
        if not all(isinstance(x, int) for x in labels):
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(np.array(labels))

        update_model_config({"num_classes":len(set(labels))})
        save_dict_labels(labels_encoded,labels)
        training_data = {"texts":texts,"labels":labels_encoded,"max_length":max_length} 
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
            df=pd.read_csv(params.csv_file,sep='\t',header=None)
            texts=df[df.columns[1]].to_list()
            labels=df[df.columns[0]].to_list()
            predict_data={"texts":texts,"labels":labels}
        elif params.sentence: 
            pass 
        else: 
            raise Exception("No Data provided for prediction. Use one of three arguments : --txt_file --csv_file or --sentence")
        
        return cls(**predict_data)


class CustomTransformerData(Dataset):
    def __init__(self,tokenizer,texts, labels, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.targets = labels
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text = " ".join(text.split())

        inputs = self.tokenizer(text,
                                max_length=self.max_len,
                                padding='max_length',
                                truncation=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
        }


class TransformerPreprocessing():
    TRAIN_BATCH_SIZE=4
    VALID_BATCH_SIZE=1
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    valid_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    def __init__(self,params):
        data = PrepareData.for_transformer(params)
        self.texts = data.texts
        self.labels = data.labelss
        self.max_length = data.max_length
        self.model_name = "/".join(params.model.split("/")[1:])
        self.tokenizer =AutoTokenizer.from_pretrained(self.model_name)

    def get_training_set(self):
        x_train, x_val, y_train, y_val=split_data(self.texts,self.labels)
        y_train = Preprocessing.convert_label(self.labels,y_train)
        y_val = Preprocessing.convert_label(self.labels, y_val)
        training_set = CustomTransformerData(self.tokenizer,x_train,y_train, self.max_length)
        validation_set = CustomTransformerData(self.tokenizer,x_val,y_val, self.max_length)
        training_loader = DataLoader(training_set, **self.train_params)
        validation_loader = DataLoader(validation_set, **self.valid_params)
        return training_loader, validation_loader
    
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
    def for_transformer_training(cls,data,model,tokenizer) -> "Preprocessing": 
        print("Preprocessing...")
        x_train, x_val, y_train, y_val=split_data(data.texts,data.labels)
        max_length = data.max_length
        inputs_train = tokenizer(x_train,return_tensors="pt",padding="max_length",max_length=max_length)
        inputs_val = tokenizer(x_val,return_tensors="pt",padding="max_length",max_length=max_length)
        x_train = model(**inputs_train)
        x_train = x_train[0].detach().cpu().numpy()
        x_val = model(**inputs_val)
        x_val = x_val[0].detach().cpu().numpy()
        y_train = cls.convert_label(data.labels,y_train)
        y_val = cls.convert_label(data.labels,y_val)
        training_data = {"x_train":x_train,"y_train":y_train,"x_val":x_val,"y_val":y_val}
        return cls(**training_data)
        
    @classmethod 
    def for_transformer_training_bis(cls,data,tokenizer) -> "Preprocessing": 
        x_train, x_val, y_train, y_val=split_data(data.texts,data.labels)
        max_length = data.max_length
        input_ids_train, attention_mask_train = cls.encode_transformer_data(tokenizer, x_train, max_length)
        input_ids_val, attention_mask_val = cls.encode_transformer_data(tokenizer, x_val, max_length)       
        x_train = {"input_ids":input_ids_train, "attention_mask":attention_mask_train}
        x_val = {"input_ids":input_ids_val, "attention_mask":attention_mask_val}
        y_train = cls.convert_label(data.labels,y_train)
        y_val = cls.convert_label(data.labels,y_val)
        training_data = {"x_train":x_train,"y_train":y_train,"x_val":x_val,"y_val":y_val}
        return cls(**training_data)

    
    @classmethod 
    def for_transformer_prediction(cls,data,tokenizer) -> "Preprocessing": 
        x_predict, y_predict = data.texts,data.labels
        max_length = data.max_length
        input_ids_predict, attention_mask_predict = cls.encode_transformer_data(tokenizer, x_predict, max_length)
        x_predict = {"input_ids":input_ids_predict, "attention_mask":attention_mask_predict}
        predict_data = {"x_predict":x_predict,"y_predict":y_predict}
        return cls(**predict_data)
    
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
        texts = data.texts
        labels = data.labels
        print(labels)
        word_index=load_word_index() 
        x_predict=np.array([cls.text_to_sequence(text) for text in texts])
        config=load_json('aa/ressources/config/model_config.json')
        x_predict = pad_sequences(x_predict, padding='post', maxlen=config['max_length'])

        labelsEncoded_labels=load_dict_labels()
        print(labelsEncoded_labels)
        #Index out of label 
        idx_ool= len(labelsEncoded_labels) + 1
        labels_labelsEncoded={v:k for k,v in labelsEncoded_labels.items()}
        y_predict=[labels_labelsEncoded.get(label) if label in labels_labelsEncoded else idx_ool for label in labels]
        y_predict = cls.convert_label(list(labelsEncoded_labels.keys()),y_predict)
        predict_data={"x_predict":x_predict,"y_predict":y_predict}
        return cls(**predict_data) 

    @classmethod
    def for_prediction_bis(cls,data):
        texts = data.texts
        word_index=load_word_index() 
        sentences=[]       
        for d in texts: 
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
    def encode_transformer_data(tokenizer, texts, max_length): 
        input_ids,attention_mask=[],[]
        for text in texts:
            inputs = tokenizer(text,padding="max_length",max_length=max_length,return_tensors='pt', truncation = True)
            input_ids.append(inputs.input_ids)
            attention_mask.append(inputs.attention_mask)
        return torch.stack(input_ids),torch.stack(attention_mask)

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
    def convert_label(labels:List,y:List):
        y = to_categorical(y,num_classes=len(set(labels)))
        return y 
    
    @staticmethod
    def text_to_sequence(text:List):
        word_index=load_word_index()
        sequence=[word_index.get(word) if word in word_index else 0 for word in text.split()]
        return np.array(sequence)

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
    
    

    

