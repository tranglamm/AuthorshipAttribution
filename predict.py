from aa import CNN
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
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_word(index):
    word_index = load_word_index() 
    index_word = {v: k for k, v in word_index.items()}
    word = index_word[index]
    return word

def sequence_to_text(sequence):
    """
    Transform a sequence of intergers into a list of text associated with its deconv value
    """
    words = []
    for index in sequence: 
        if index == 0: 
            word = "PAD"
        else: 
            word = get_word(index)
        
        words.append(word)
    return words

def get_parser():
    """
    Generate a parameters parser.
    
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Evaluating model")
    requiredNamed = parser.add_argument_group('required named arguments')
    
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
    data = PrepareData.for_prediction(params)
    preprocessing=Preprocessing.for_prediction(data)
    predict_data = preprocessing.x_predict
    print("Total of predict data: ",len(predict_data))
    model = load_model(params.model)
    predictions = model.predict(predict_data)

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
    
    deconv = deconv_model.predict(predict_data)
    print("DECONVOLUTION SHAPE : ", deconv.shape)
    
    result = []

    for sequence, deconv_values, prediction in zip(predict_data,deconv,predictions):
        sentence = {}
        sentence["prediction"] = prediction.tolist()
        words=sequence_to_text(sequence)
        sentence["sentence"]= [(word,float(np.sum(deconv_value))) for word,deconv_value in zip(words,deconv_values)]
        result.append(sentence)
        
    """
    for sentence_nb in range(len(predict_data)):
        sentence = {}
        sentence["sentence"] = ""
        sentence["prediction"] = predictions[sentence_nb].tolist()
        for i in range(len(predict_data[sentence_nb])):
            word = ""
            index = predict_data[sentence_nb][i]
            if index == 0: 
              word = "PAD"
            else: 
              word = get_word(index)
            if word == "PAD":
                word = "PAD"

			# READ DECONVOLUTION 
            deconv_value = float(np.sum(deconv[sentence_nb][i]))
            sentence["sentence"] += word + ":" + str(deconv_value) + " "
        result.append(sentence)
    """
    print("----------------------------")
    results_dir="aa/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result_path = os.path.join(results_dir, params.txt_file.split('/')[-1] + ".res")
    with open(result_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(result,cls=NumpyEncoder,ensure_ascii=False,indent=2))

if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)
