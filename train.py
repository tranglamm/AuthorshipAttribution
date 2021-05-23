from aa import CNN
from aa.config_utils import *
from aa import preprocessing
import tensorflow
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Conv2D
import os
import json
import random
import argparse
import time
import numpy as np
from aa.preprocessing import *
from aa.embedder import *
import logging
logging.basicConfig(level=logging.DEBUG)

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

    parser.add_argument("--val_data", type=str, default="",
                        help="Path of validation file CSV. If not specified, we will split training data into train_data and val_data")

    parser.add_argument("--lg", type=str, default="",
                        help="Choose a language to download FastText pretrained Word Embeddings")

    parser.add_argument("--reduce_dim", type=int, default="300",
                        help="Reduce dimension of FastText pretrained Word Embeddings")

    parser.add_argument("--custom_emb", type=str, default="w2v",
                        help="You can choose between w2v or ft to train your own embeddings")
    """
    parser.add_argument("--model_config", type=str, default="aa/config/model_config.json",
                        help="Path of file training_config.json")       

    parser.add_argument("--emb_config", type=str, default="aa/config/emb_config.json",
                        help="Path of file training_config.json") 
    """
    requiredNamed.add_argument("--output_file", type=str, default="aa/models/trained_models/CnnModel",
                        help="Path of model", required=True)
    return parser 

def main(params):
    preprocessing=Preprocessing(params)
    x_train, y_train = preprocessing.get_training_data()
    x_val, y_val = preprocessing.get_validation_data()
    texts = preprocessing.get_texts()
    print(preprocessing.get_vocab_size())
    print(preprocessing.get_vocab())
    print(x_train.shape)
    print(x_val.shape)
    #Load embedding config 
    emb_config = EmbConfig.from_json_file("aa/config/emb_config.json")
    #Load embeddings
    #if have params lg => load pretrained embedding from FastText 
    if params.lg: 
        load_pretrained_embeddings(params)
        embedding_matrix=preprocessing.prepare_txt_embedding(emb_config)
    elif params.custom_emb: 
        if params.custom_emb=="w2v":
            custom_w2v_embeddings(texts,emb_config)
        elif params.custom_emb=="ft":
            custom_ft_embeddings(texts,emb_config)
        else: 
            print("You have to choose between w2v or ft")
        embedding_matrix=preprocessing.prepare_custom_embedding(emb_config)
    #Build model 
    model_config = ModelConfig.from_json_file("aa/config/model_config.json")
    try: 
        assert model_config.emb_dim == emb_config.vector_size
    except AssertionError: 
        raise AssertionError("The embedding dimension of model and word embeddings are not equal")   
    else:  
        if not os.path.exists("aa/output_models"):
            os.makedirs("aa/output_models")
        output_file=params.output_file
        CnnModel=CNN(model_config,weight=embedding_matrix)
        model,deconv_model=CnnModel.get_model()
    
        mc=[ModelCheckpoint(output_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
            #EarlyStopping(patience=2,monitor="val_accuracy")
            ]
        history=model.fit(x_train, y_train,
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

if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    print(params.train_data)
    main(params)
