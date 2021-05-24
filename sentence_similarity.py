# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
from sentence_transformers import *
from sentence_transformers import models
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
import scipy.spatial
from torch import nn 

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
logging.basicConfig(level=logging.DEBUG)

def get_parser():
    """
    Generate a parameters parser.
    
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Training sentence similarity model")
    requiredNamed = parser.add_argument_group('required named arguments')
    # data
    requiredNamed.add_argument("--train_data", type=str, default="",
                        help="Path of training file CSV", required=True)

    requiredNamed.add_argument("--query", type=str, default="",
                        help="Path of output", required=True)

    
    parser.add_argument("--sep", type=str, default="\t",
                        help="Define Separator of your csv file")

    parser.add_argument("--lg", type=str, default="",
                        help="Choose a language to compute sentence similarity. We support only for this moment French (fr) and English (en)")
    """
    parser.add_argument("--lg", type=str, default="",
                        help="Choose a language to download FastText pretrained Word Embeddings")

    parser.add_argument("--reduce_dim", type=int, default="300",
                        help="Reduce dimension of FastText pretrained Word Embeddings")

    parser.add_argument("--custom_emb", type=str, default="w2v",
                        help="You can choose between w2v or ft to train your own embeddings")
    
    parser.add_argument("--val_data", type=str, default="",
                        help="Path of validation file CSV. If not specified, we will split training data into train_data and val_data")

    parser.add_argument("--model_config", type=str, default="aa/config/model_config.json",
                        help="Path of file training_config.json")       

    parser.add_argument("--emb_config", type=str, default="aa/config/emb_config.json",
                        help="Path of file training_config.json") 
    """
    
    return parser 

def main(params):
    if params.lg =='fr':
        model_transformer='camembert-base'
    elif params.lg =='en':
        model_transformer='paraphrase-distilroberta-base-v1'  
    else: 
        raise Exception (f"{params.lg} Not support for this moment")

    # Use CamemBERT for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_transformer,max_seq_length=50)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if params.train_data and params.query: 
        data=PrepareData.for_training(params)
        preprocessing=Preprocessing.for_training(data)
        x_train, y_train = preprocessing.x_train, preprocessing.y_train
        x_val, y_val = preprocessing.x_val, preprocessing.y_val

        
        corpus_embeddings = model.encode(data.texts,batch_size=16,show_progress_bar=True)


        query = params.query
        query_embedding = model.encode(query)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = 5
        #distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        distances=cos(query_embedding, corpus_embeddings)[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        labelsEncoded_labels=load_dict_labels()
        for idx, distance in results[0:closest_n]:
            print(data.texts[idx].strip(),  labelsEncoded_labels.get(data.labels[idx]), "(Score: %.4f)" % (1-distance))

    elif params.query1 and params.query2: 
        embedding1 = model.encode(params.query1, convert_to_tensor=True)
        embedding2 = model.encode(params.query2, convert_to_tensor=True)
        cosine_scores = cos(embedding1, embedding2)
        print(cosine_scores)

if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)