from aa import CNN, config
from aa.config_utils import *
from aa.preprocessing import *
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
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
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def get_parser():
    """
    Generate a parameters parser.
    
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Training model")
    requiredNamed = parser.add_argument_group('required named arguments')
    
    # data
    """
    requiredNamed.add_argument("--train_data", type=str, default="",
                        help="Path of training file CSV", required=True)

    parser.add_argument("--val_data", type=str, default="", 
                        help="Path of validation file CSV. If not specified, we will split training data into train_data and val_data")
    """
    requiredNamed.add_argument("--file_vec", type=str, default="",required=True,
                        help="Path of your Word Embeddings that can be found in aa/pretrained_emb directory")

    requiredNamed.add_argument("--model", type=str, default="",required=True,
                        help="Path of your model that can be found in trained_models/--output_file dir")

    parser.add_argument("--sentence", type=str, default="",
                        help="Sentence to predict")

    parser.add_argument("--txt_file", type=str, default="",
                        help="Path of txt file")
    
    return parser 

def main(params):
    result = []

    preprocessing=Preprocessing(params)
    texts = preprocessing.get_predict_data()
    print(texts[:2])
    model = load_model(params.model)
    predictions = model.predict(texts)
    #print(predictions)	
    print("----------------------------")
    print("DECONVOLUTION")
    print("----------------------------")
    
    deconv_model = load_model(params.model + ".deconv")

    for layer in deconv_model.layers:	
        if type(layer) is Conv2D:
            deconv_weights = layer.get_weights()[0]
            print(deconv_weights.shape)
    
    print(deconv_model.layers[-1].get_weights()[0].shape)
    deconv_bias = deconv_model.layers[-1].get_weights()[1]
    deconv_model.layers[-1].set_weights([deconv_weights, deconv_bias])
    
    deconv = deconv_model.predict(texts)
    print("DECONVOLUTION SHAPE : ", deconv.shape)
    
    #my_dictionary = preprocessing.get_vocab()
    print(len(texts))
    
    for sentence_nb in range(len(texts)):
        sentence = {}
        sentence["sentence"] = ""
        sentence["prediction"] = predictions[sentence_nb].tolist()
        for i in range(len(texts[sentence_nb])):
            word = ""
            index = texts[sentence_nb][i]
            if index == 0: 
              word = "PAD"
            else: 
              word = preprocessing.get_word(index)
            if word == "PAD":
                word = "PAD"

			# READ DECONVOLUTION 
            deconv_value = float(np.sum(deconv[sentence_nb][i]))
            sentence["sentence"] += word + ":" + str(deconv_value) + " "
        result.append(sentence)
    
    print("----------------------------")
    results_dir="aa/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_path = os.path.join(results_dir, params.txt_file.split('/')[-1] + ".res")
    with open(result_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(result,ensure_ascii=False))

if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)
