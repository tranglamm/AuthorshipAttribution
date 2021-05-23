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

    requiredNamed.add_argument("--output", type=str, default="",
                        help="Path of output", required=True)

    parser.add_argument("--sep", type=str, default="\t",
                        help="Define Separator of your csv file")

    parser.add_argument("--lg", type=str, default="",
                        help="Choose a language to download FastText pretrained Word Embeddings")

    parser.add_argument("--reduce_dim", type=int, default="300",
                        help="Reduce dimension of FastText pretrained Word Embeddings")

    parser.add_argument("--custom_emb", type=str, default="w2v",
                        help="You can choose between w2v or ft to train your own embeddings")
    """
    parser.add_argument("--val_data", type=str, default="",
                        help="Path of validation file CSV. If not specified, we will split training data into train_data and val_data")

    parser.add_argument("--model_config", type=str, default="aa/config/model_config.json",
                        help="Path of file training_config.json")       

    parser.add_argument("--emb_config", type=str, default="aa/config/emb_config.json",
                        help="Path of file training_config.json") 
    """
    
    return parser 

def main(params):
    data=PrepareData.for_training(params)
    preprocessing=Preprocessing.for_training(data)
    x_train, y_train = preprocessing.x_train, preprocessing.y_train
    x_val, y_val = preprocessing.x_val, preprocessing.y_val
    print(preprocessing.get_vocab_size())
    print(preprocessing.get_vocab())
    print(x_train.shape)
    print(x_val.shape)

    #Load embedding config 
    emb_config = EmbConfig.from_json_file("aa/ressources/config/emb_config.json")
    #Load embeddings
    #if have params lg => load pretrained embedding from FastText 
    if params.lg: 
        load_pretrained_embeddings(params)
        embedding_matrix=preprocessing.prepare_txt_embedding(params.lg,emb_config)

    elif params.custom_emb: 
        if params.custom_emb=="w2v":
            custom_w2v_embeddings(data.texts,emb_config)
        elif params.custom_emb=="ft":
            custom_ft_embeddings(data.texts,emb_config)
        else: 
            print("You have to choose between w2v or ft")
        embedding_matrix=preprocessing.prepare_custom_embedding(emb_config)

    #Build model 
    model_config = ModelConfig.from_json_file("aa/ressources/config/model_config.json")
    if model_config.emb_dim != emb_config.vector_size:
        raise Exception("The embedding dimension of model and word embeddings are not equal")   
    else:  
        output_dir="aa/output_models"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file=os.path.join(output_dir,params.output)
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
    main(params)
