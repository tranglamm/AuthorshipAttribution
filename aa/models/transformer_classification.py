from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf 
from tensorflow.keras.layers import Input
from transformers import AutoTokenizer, pipeline,AutoModelForSequenceClassification, AutoModel
import numpy as np 
import torch
from torch import nn 
import torch.nn.functional as F

NB_FILTERS = 100
FILTER_SIZES =[3]

class AutoModelTransformers(nn.Module):
    def __init__(self,config,pretrained_model_name):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name,num_labels = config.num_labels)
    def forward(self, 
                **inputs
                ):
        outputs = self.classifier(**inputs)
        return outputs

class ClassificationMLP(nn.Module):
    """
    TODO
    """
    def __init__(self,config,pretrained_model_name,static:bool):
        super().__init__()
        self.static = static
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.batchnorm = nn.BatchNorm1d(config.hidden_size)
        self.pre_classifier = nn.Linear(config.hidden_size,config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, 
                **inputs
                ):
        if self.static: 
            with torch.no_grad():
                outputs = self.model(**inputs)
        else: 
            outputs = self.model(**inputs)
        last_hidden_state = outputs[0]
        pooler = last_hidden_state[:,0,:]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        return logits



class CombineCnn(nn.Module):
    """
    TODO
    Review Config
    """
    filter_sizes = [3]
    def __init__(self,config,pretrained_model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = config.nb_filters, 
                                              kernel_size = (fs, config.hidden_size)) 
                                    for fs in self.filter_sizes
                                    ])
        self.dropout = nn.Dropout(0.2)
        self.pre_classifier = torch.nn.Linear(len(self.filter_sizes)*config.nb_filters,100)
        self.classifier = torch.nn.Linear(100, config.num_labels)


    def forward(self, 
                **inputs
                ):
       
        outputs= self.model(**inputs)

        #last_hidden_state.size() = batch * 62 * 78      
        last_hidden_state = outputs[0]
        

        #embedded.size() = batch,1 * 62 * 768
        embedded = last_hidden_state.unsqueeze(1)
        
        #conved.size() = batch * 512 * 60 * 1
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        #pooled.size() = batch * 512
        pooled = [F.max_pool1d(F.relu(conv), conv.shape[2]).squeeze(2) for conv in conved]


        cat = self.dropout(torch.cat(pooled, dim = 1))

        pre_classifier = F.relu(self.pre_classifier(cat))
        logits = self.classifier(pre_classifier)
        return logits


class Transformers: 
    def __init__(self,**kwargs):
        pass

    @classmethod
    def classification_1(cls,config):
        input = tf.keras.layers.Input(shape=(config.max_position_embeddings,config.hidden_size))
        X = input[:,0,:]
        X = tf.keras.layers.Dense(4, activation='softmax')(X)
        model = tf.keras.Model(inputs=input, outputs = X)

        model.summary()
        return model

    @classmethod
    def classification_2(cls,config):
        input = tf.keras.layers.Input(shape=(config.max_position_embeddings,config.hidden_size))
        cls_token = input[:,0,:]
        X = tf.keras.layers.BatchNormalization()(cls_token)
        X = tf.keras.layers.Dense(config.hidden_size, activation='relu')(X)
        X = tf.keras.layers.Dropout(config.dropout)(X)
        X = tf.keras.layers.Dense(config.num_labels, activation='softmax')(X)
        model = tf.keras.Model(inputs=input, outputs = X)
        
        model.summary()
        
        return model
    
    def combine_lstm(cls,config):
        input= tf.keras.layers.Input(shape=(config.max_position_embeddings,config.hidden_size))
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(input)
        X = tf.keras.layers.GlobalMaxPool1D()(X)
        X = tf.keras.layers.Dense(50, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(config.num_labels, activation='sigmoid')(X)
        model = tf.keras.Model(inputs=input, outputs = X)
        
        model.summary()
        
        return model


    def combine_cnn(cls,config,filter_sizes=[2,3,4]):
        input= tf.keras.layers.Input(shape=(config.max_position_embeddings,config.hidden_size))
        convs = []
        for fsz in filter_sizes:
            X = tf.keras.layers.Conv1D(100, fsz, activation='relu',padding='same')(input)
            convs.append(X)
            
        X = tf.keras.layers.Concatenate(axis=-1)(convs)
        X = tf.keras.layers.GlobalMaxPooling1D()(X)
        X = tf.keras.layers.Dense(50, activation='relu')(X)
        X = tf.keras.layers.Dropout(0.5)(X)
        X = tf.keras.layers.Dense(config.num_labels, activation='sigmoid')(X)
        
        model = tf.keras.Model(inputs=input, outputs = X)
        model.summary()
    
    @classmethod
    def combine_cnn_deconv(cls,config):
        conv_array = []
        maxpool_array = []
        deconv_array = []

        input = tf.keras.layers.Input(shape=(config.max_position_embeddings,config.hidden_size))
        filter_sizes = [2,3]
        reshape = tf.keras.layers.Reshape((config.max_position_embeddings,config.hidden_size,1))(input)
        for filter in filter_sizes:
            conv = tf.keras.layers.Conv2D(100, (filter, config.hidden_size), padding='valid')(reshape)
            conv_array.append(conv)
            max_pool =  tf.keras.layers.GlobalMaxPooling2D()(conv)	
            maxpool_array.append(max_pool)
            deconv = tf.keras.layers.Conv2DTranspose(1,(filter, config.hidden_size))(conv)
            deconv_array.append(deconv)

        concat = tf.keras.layers.Concatenate(axis=-1)(maxpool_array)
        deconv_model = tf.keras.Model(inputs=input, outputs=deconv_array)
        deconv_model.summary()

        dropout = tf.keras.layers.Dropout(0.2)(concat)

        hidden_dense = tf.keras.layers.Dense(128,activation='relu')(dropout)
        output = tf.keras.layers.Dense(config.num_labels, activation='softmax')(hidden_dense)

        # this creates a model that includes
        model= tf.keras.Model(inputs=input, outputs=output)
        model.summary()

        
