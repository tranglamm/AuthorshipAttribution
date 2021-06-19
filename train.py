"""
@author Trang Lam - github.com/tranglamm

"""
from aa.config_utils import TrainingConfig
from tensorflow.python.autograph.pyct import transformer
from aa.models.transformer_classification import CombineCnn, Transformers
from tensorflow.python.keras.preprocessing.text import Tokenizer
from aa import *

import tensorflow
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
import random
import argparse
import time
import numpy as np
import torch 
import logging

def get_parser():
    """
    Generate a parameters parser.
    
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Training model")
    requiredNamed = parser.add_argument_group('required named arguments')
    # data
    requiredNamed.add_argument("--train_data", type=str, default="",
                        help="Path of training file CSV", required=True)

    parser.add_argument("--max_length", type=int,
                        help="Define the maximum sequence length --> it will be used to truncate and pad the sequence.")

    requiredNamed.add_argument("--output", type=str, default="",
                        help="Path of output", required=True)

    parser.add_argument("--sep", type=str, default="\t",
                        help="Define Separator of your csv file")

    parser.add_argument("--lg", type=str, default="",
                        help="Choose a language to download FastText pretrained Word Embeddings")

    parser.add_argument("--model", type=str,default="cnn",
                        help=r"""Choose a model you want to use for training (cnn, attention, transformers). 
                        If transformers, please specify the pretrained_model_name. For example FlauBERT --model transformers/flaubert/flaubert-cased """)

    parser.add_argument("--reduce_dim", type=int, default="300",
                        help="Reduce dimension of FastText pretrained Word Embeddings")

    parser.add_argument("--custom_emb", type=str, default="w2v",
                        help="You can choose between w2v or ft to train your own embeddings")
    
    parser.add_argument("--val_data", type=str, default="",
                        help="Path of validation file CSV. If not specified, we will split training data into train_data and val_data")
    """
    parser.add_argument("--model_config", type=str, default="aa/config/model_config.json",
                        help="Path of file training_config.json")       

    parser.add_argument("--emb_config", type=str, default="aa/config/emb_config.json",
                        help="Path of file training_config.json") 
    """
    
    return parser 

def main(params):
    output_dir="aa/output_models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file=os.path.join(output_dir,params.output)

    if params.model == "cnn" or params.model == "attention": 
        data=PrepareData.for_training(params)
        preprocessing=Preprocessing.for_training(data)
        x_train, y_train = preprocessing.x_train, preprocessing.y_train
        x_val, y_val = preprocessing.x_val, preprocessing.y_val
        print(preprocessing.get_vocab_size())
        print(preprocessing.get_vocab(data))
        print(x_train.shape)
        print(x_val.shape)

        #Load config_json for embedding and model 
        emb_config = EmbConfig.from_json_file("aa/ressources/config/emb_config.json")
        model_config = ModelConfig.from_json_file("aa/ressources/config/model_config.json")
        #Load embeddings
        #if have params lg => load pretrained embedding from FastText 
        if params.lg: 
            load_pretrained_embeddings(params)
            embedding_matrix=preprocessing.prepare_txt_embedding(data,params.lg,emb_config)

        elif params.custom_emb: 
            if params.custom_emb=="w2v":
                custom_w2v_embeddings(data.texts,emb_config)
            elif params.custom_emb=="ft":
                custom_ft_embeddings(data.texts,emb_config)
            else: 
                print("You have to choose between w2v or ft")
            embedding_matrix=preprocessing.prepare_custom_embedding(data,emb_config)

        #Assert emb_dim in model_config.json == emb_config.json 
        if model_config.emb_dim != emb_config.vector_size:
            raise Exception("The embedding dimension of model and word embeddings are not equal")   
        
        else:  
            opt = optimizers.Adam(lr=model_config.lr)
            mc=[ModelCheckpoint(output_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
                #EarlyStopping(patience=2,monitor="val_accuracy")
                ]
            if params.model=="cnn":
                CnnModel=CNN(model_config,weight=embedding_matrix)
                model,deconv_model=CnnModel.get_model()
                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(x_train, y_train,
                            epochs=model_config.num_epochs,
                            validation_data=(x_val,y_val),
                            batch_size=model_config.batch_size,
                            callbacks=mc)

                model.save(output_file)

                #save deconv model
                i = 0
                for layer in model.layers:	
                    weights = layer.get_weights()
                    deconv_model.layers[i].set_weights(weights)
                    i += 1
                    if type(layer) is Conv2D:
                        break
                deconv_model.save(output_file + ".deconv")

            elif params.model=="attention":
                model,attention_model = Attention.build(model_config)
                model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(x_train, y_train,
                            epochs=model_config.num_epochs,
                            validation_data=(x_val,y_val),
                            batch_size=model_config.batch_size,
                            callbacks=mc)
                attention_model.save(output_file + ".attention")
                model.save(output_file)

    elif len(params.model.split("/")) >1:
        
        pretrained_model_name="/".join(params.model.split("/")[1:])
        transformer_config=CONFIG_JSON["transformer_config"]
        config = AutoConfig.from_pretrained(transformer_config)
        
        #Preprocessing Data 
        preprocessing=TransformerPreprocessing(params)
        training_loader,validation_loader = preprocessing.get_training_set()
        
        #Build Model
        
        model = CombineCnn(config,pretrained_model_name)
        
        #Train model
        training_config_json = CONFIG_JSON["training_config"]
        training_config = TrainingConfig.from_json_file(training_config_json)
        training = TrainingTransformer(model,training_config)
        training.train(training_loader)

        #Save model
        output_file_name = output_file+".pt"
        torch.save(model.state_dict(), output_file_name)


if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)
