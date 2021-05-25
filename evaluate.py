"""
Evaluate a csv file => 
    Matrix Confusion 
    CSV file : true_target predict_target prediction_score 
"""
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
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import *
from IPython.display import display, HTML


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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

def draw_confusion_matrix(y_predict,Y_predict):
    """
    y_predict : true labels
    Y_predict : labels predicted by model
    """
    matrix = confusion_matrix(y_predict, Y_predict)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    fmt = "%.2f"
    labelsEncoded_labels =load_dict_labels()
    df_cm = pd.DataFrame(matrix, columns=list(labelsEncoded_labels.values()), index = list(labelsEncoded_labels.values()))
    df_cm.index.name = 'Vraies classes'
    df_cm.columns.name = 'Classes prédites'
    plt.figure(figsize=(10,8))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,fmt=".2f", annot_kws={"size": 14}) # font size
    plt.show()

def colorize(words,deconv_values,prediction):


    #word_cmap = matplotlib.cm.PiYG
    #word_cmap = matplotlib.cm.BuPu
    word_cmap = matplotlib.cm.GnBu
    #prob_cmap = matplotlib.cm.Pastel
    template = '<span class="barcode"; style="color: black; background-color: {}">{} </span>'
    colored_string = ''
    # Use a matplotlib normalizer in order to make clearer the difference between values
    normalized_and_mapped = matplotlib.cm.ScalarMappable(cmap=word_cmap).to_rgba(deconv_values)
    for word, color in zip(words, normalized_and_mapped):
        color = matplotlib.colors.rgb2hex(color[:3])
        colored_string += template.format(color, word)
    color="#B5B3D5"
    colored_string += template.format(color, "    Label: {} |".format(np.argmax(prediction)))
    
    prob = np.amax(prediction)
    #color = matplotlib.colors.rgb2hex(prob_cmap(prob)[:3])
    
    colored_string += template.format(color, "{:.2f}%".format(prob*100)) + '|'
    
    return colored_string

def colorizeAttention(words,attention_scores,prediction):


    #word_cmap = matplotlib.cm.PiYG
    #word_cmap = matplotlib.cm.BuPu
    word_cmap = matplotlib.cm.BuPu
    #prob_cmap = matplotlib.cm.Pastel
    template = '<span class="barcode"; style="color: black; background-color: {}">{} </span>'
    colored_string = ''
    # Use a matplotlib normalizer in order to make clearer the difference between values
    normalized_and_mapped = matplotlib.cm.ScalarMappable(cmap=word_cmap).to_rgba(attention_scores)
    print(normalized_and_mapped.shape)
    for word, color in zip(words, normalized_and_mapped):
        color = matplotlib.colors.rgb2hex(color[:3])
        colored_string += template.format(color, word)
    color="#B5B3D5"
    colored_string += template.format(color, "    Label: {} |".format(np.argmax(prediction)))
    
    prob = np.amax(prediction)
    #color = matplotlib.colors.rgb2hex(prob_cmap(prob)[:3])
    
    colored_string += template.format(color, "{:.2f}%".format(prob*100)) + '|'
    
    return colored_string


def get_parser():
    """
    Generate a parameters parser.
    
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Training model")
    requiredNamed = parser.add_argument_group('required named arguments')
    
    parser.add_argument("--file_vec", type=str, default="",
                        help="Path of your Word Embeddings that can be found in aa/pretrained_emb directory")

    requiredNamed.add_argument("--model", type=str, default="",required=True,
                        help="Path of your model that can be found in trained_models/--output_file dir")

    parser.add_argument("--sentence", type=str, default="",
                        help="Sentence to predict")

    parser.add_argument("--txt_file", type=str, default="",
                        help="Path of txt file")

    parser.add_argument("--csv_file", type=str, default="",
                        help="Path of csv file")
    
    parser.add_argument("--output_csv", type=str, default="",
                        help="Path of csv file")
    
    parser.add_argument("--attention", type=bool, default="",
                        help="if you want to visualize attention score ")

    parser.add_argument("--deconv", type=bool, default="",
                        help="Output a file res with the activation score given by the deconvolution for each word")
    return parser 

def main(params):
    data = PrepareData.for_prediction(params)
    preprocessing=Preprocessing.for_prediction(data)
    x_predict = preprocessing.x_predict
    y_predict = preprocessing.y_predict
    print("Total of predict data: ",len(x_predict))
    model = load_model(params.model)
    predictions = model.predict(x_predict)
    Y_predict=np.argmax(predictions,axis=1)
    print(Y_predict)
    print("----------------------------")
    loss, accuracy = model.evaluate(x_predict, y_predict, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    print("----------------------------")
    print(np.argmax(y_predict,axis=1))
    report = classification_report(np.argmax(y_predict,axis=1),Y_predict)
    print(report)
    draw_confusion_matrix(np.argmax(y_predict,axis=1),Y_predict)
    
    if params.output_csv: 
        """
        Output a csv file consits of: 
            Text, True Label, Predict Label, Predictions Score 
        """
        results_dir="aa/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        output_data={"texts":data.texts,
                    "true_labels":data.labels,
                    "predicted_labels": Y_predict,
                    "predictions_score": predictions
                    }
        df = pd.DataFrame(output_data)
        file_name=os.path.join(results_dir, params.output_csv)
        df.to_csv(file_name,index=False)

    if params.deconv:
        print("----------------------------")
        print("DECONVOLUTION")
        print("----------------------------")
        
        deconv_model = load_model(params.model + ".deconv")

        for layer in deconv_model.layers:	
            if type(layer) is Conv2D:
                deconv_weights = layer.get_weights()[0]
                #print(deconv_weights.shape)
        
        #print(deconv_model.layers[-1].get_weights()[0].shape)
        deconv_bias = deconv_model.layers[-1].get_weights()[1]
        deconv_model.layers[-1].set_weights([deconv_weights, deconv_bias])
        
        deconv = deconv_model.predict(x_predict)
        print("DECONVOLUTION SHAPE : ", deconv.shape)
        
        result = []

        for sequence, deconv_values, label, prediction in zip(data.texts,deconv,data.labels,predictions):
            sentence = {}
            sentence["prediction"] = prediction.tolist()
            words=sequence.split()
            sentence["sentence"]= [(word,float(np.sum(deconv_value))) for word,deconv_value in zip(words,deconv_values)]
            result.append(sentence)
            deconv_values = [float(np.sum(deconv_value)) for deconv_value in deconv_values]
            colored_string=colorize(words,deconv_values,prediction)
            print(colored_string)
            display(HTML(colored_string))
            
        print("----------------------------")
        results_dir="aa/results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        result_path = os.path.join(results_dir, params.txt_file.split('/')[-1] + ".res")
        with open(result_path, "w", encoding='utf-8') as f:
            f.write(json.dumps(result,cls=NumpyEncoder,ensure_ascii=False,indent=2))

    if params.attention: 
        attention_model = load_model(params.model + ".attention")
        atn_scores= attention_model.predict(x_predict)
        for text, attention_scores, prediction in zip(data.texts,atn_scores,predictions):
            words=text.split()
            attention_scores=[float(atn_score) for atn_score in attention_scores]
            print(attention_scores)
            colored_string=colorizeAttention(words,attention_scores,prediction)
            print(colored_string)

if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)
