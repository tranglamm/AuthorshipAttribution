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

def read_csv_file(data_path,clean_text:bool=True):
    if Path(data_path).exists():
        df=pd.read_csv(data_path,sep="\t",encoding="utf-8")
    else: 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    texts=df.columns[1]
    labels=df.columns[0]
    df = df.dropna()
    if clean_text:
        df[texts]=df[texts].apply(lambda x:str(x)).apply(clean_tweet)
    texts = df[texts].to_list()
    labels=df[labels].to_list()
    return texts, labels 

def load_json(file_json):
    with open(file_json,"r",encoding="utf-8") as f: 
        data = json.load(f)
    return data
    
def update_model_config(dict_config):
    with open("aa/config/model_config.json", "r+") as file:
        data = json.load(file)
        #if list(dict_config.keys())[0] not in data:
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)

def update_emb_config(dict_config):
    with open("aa/config/emb_config.json", "r+") as file:
        data = json.load(file)
        #if list(dict_config.keys())[0] not in data:
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)

def save_dict_labels(labels,labels_encoded):
    dict_labels=dict(zip(labels, labels_encoded))
    path= "aa/ressources/dict_labels.pickle"
    with open(path, 'wb') as handle:
        pickle.dump(dict_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_labels():
    path= "aa/ressources/dict_labels.pickle"
    if Path(path).exists():
        with open(path, 'rb') as handle:
            dict_labels = pickle.load(handle)
    else: 
        print("Please train your model before running test ...")

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

def transform_labels(y_train, y_val):
    labels= y_train + y_val
    labels_encoded = np.array(labels)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    Y_train = label_encoder.transform(y_train)
    Y_val = label_encoder.transform(y_val)
    #labels_encoded=label_encoder.fit_transform(labels_encoded)
    return Y_train, Y_val

def split_data(texts,labels):
    x_train, x_val, y_train, y_val = train_test_split(texts,labels, test_size=0.2,random_state=42)
    X=[x_train, x_val]
    Y=[y_train, y_val]
    return x_train, x_val, y_train, y_val