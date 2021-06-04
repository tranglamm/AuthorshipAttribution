"""
@author Trang Lam - github.com/tranglamm

"""
import re
import pandas as pd 
from pathlib import Path
import errno
import os
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import pickle 
import json
from .utils import *
from .file_utils import *

ressources_dir = RESSOURCES_DIR 
model_config_json = CONFIG_JSON["model_config"]

def read_csv_file(data_path,sep,clean_text:bool=False):
    r"""
    TODO: 
    can take into account a csv like column 0 : Text and column 1: Label
    Input :
        data_path = Path of csv file 
        CSV file has to be represented as : LABELS Columns, TEXT Columns - by default sep="\t" 
        => Change sep by using --sep 
        Split data according to max_length defined in model_config.json
    Output: 
        texts, labels 
    """
    if Path(data_path).exists():
        df=pd.read_csv(data_path,sep=sep,encoding="utf-8",header=None)
    else: 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    texts=df.columns[1]
    labels=df.columns[0]
    df = df.dropna()
    if clean_text:
        df[texts]=df[texts].apply(lambda x:str(x)).apply(clean_tweet)
        
    texts = df[texts].to_list()
    labels=df[labels].to_list()
    texts,labels=split_sent(texts,labels)
    return texts, labels 


def split_sent(texts, labels):
    r"""
        Split Data according to max_length defined in model_config.json
    """
    config=load_json(model_config_json)
    max_length=config['max_length']
    data = (np.zeros((len(texts),max_length ))).astype('int32')
    new_texts, new_labels = [],[]
    for text,label in zip(texts,labels):
        tokens=[tok for tok in text.split()]
        if len(tokens)>max_length:
            text=[tokens[x:x+max_length] for x in range(0, len(tokens), max_length)]
            label=[label]*len(text)
            new_texts.extend([" ".join(sub_text) for sub_text in text])
            new_labels.extend(label)
        else:
            new_texts.extend([text])
            new_labels.extend([label])
    return new_texts, new_labels



def save_word_index(word_index):
    path = os.path.join(ressources_dir,"word_index.pickle")
    with open(path, 'wb') as handle:
        pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_word_index():
    path = os.path.join(ressources_dir,"word_index.pickle")
    with open(path, 'rb') as handle:
        word_index = pickle.load(handle)
    return word_index

def save_dict_labels(labels_encoded,labels):
    dict_labels=dict(zip(labels_encoded, labels))
    path = os.path.join(ressources_dir,"dict_labels.pickle")
    with open(path, 'wb') as handle:
        pickle.dump(dict_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_labels():
    path = os.path.join(ressources_dir,"dict_labels.pickle")
    if Path(path).exists():
        with open(path, 'rb') as handle:
            dict_labels = pickle.load(handle)
    else: 
        print("Please train your model before running test ...")
    return dict_labels

def clean_tweet(texte):
    texte=re.sub(r'&gt;|&amp;|"','',texte)
    #Erreurs en collectant les données: "c?est","d?un" => Remplacer "?"  par "'" 
    texte=re.sub(r'(\w+?)\?(\w+?)',r"\1'\2",texte)
    #Remove web adresses: http:// or https://, www.blabla.com, 
    texte = re.sub(r'\w+://\S+', '', texte)
    texte = re.sub(r'www.\S+', '', texte)
    texte = re.sub(r'\w?:/+.+', '', texte)
    #text=re.sub(r"(%s+')"%(pattern_fr),r'\1 ',text) 
    #Remove RT@mention - AF Spécial pour les tweets Remove all retweet mentions. Optional.
    #text=re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+',r'',text)
    #Remove mention ?? useful or not??  - AF justifier l'utilité d'un point de vu linguistique
    #text=re.sub(r'@[\S]+',r'',text)
    #Remove punctuations
    texte=re.sub("…",' ',texte)
    texte=re.sub(r'[!"’«»$%&()*+,./:;<=>?[\]^_`{|}~]',' ',texte)
    return texte

def inverse_labels(labels_encoded, labels):
    """
    Y == labels_encoded
    y == labels_orig
    """
    return dict(zip(labels_encoded, labels))


def split_data(texts,labels):
    #x_train, x_val, y_train, y_val = train_test_split(texts,labels, test_size=0.1,random_state=42)
    #X=[x_train, x_val]
    #Y=[y_train, y_val]
    return train_test_split(texts,labels, test_size=0.1,random_state=42)