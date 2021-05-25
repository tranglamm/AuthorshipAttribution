from aa.models import attention
from aa import CNN, Attention
from aa.config_utils import *
from aa import preprocessing
import tensorflow
from tensorflow import keras
from tensorflow.keras import optimizers
import umap
import hdbscan
from umap import plot
from bokeh.plotting import show, save, output_notebook, output_file
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

from transformers import FlaubertTokenizer, FlaubertModel
import torch
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range,).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Document
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Document": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes



def get_parser():
    """
    Generate a parameters parser.
    
    """
    parser = argparse.ArgumentParser(description="Visualize Data")
    requiredNamed = parser.add_argument_group('required named arguments')
    
    requiredNamed.add_argument("--train_data", type=str, default="",required=True,
                        help="Path of training file CSV")
                        
    parser.add_argument("--sep", type=str, default="\t",
                        help="")
    """
    requiredNamed.add_argument("--model", type=str, default="",required=True,
                        help="Path of your model that can be found in aa/output_models/--output_file dir")
    """

    parser.add_argument("--lg", type=str, default="",
                        help="Language of your data")

    return parser

def main(params):
    tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')
    model = FlaubertModel.from_pretrained('flaubert/flaubert_base_cased')
    

    data=PrepareData.for_training(params)
    preprocessing=Preprocessing.for_training(data)
    x_train, y_train = preprocessing.x_train, preprocessing.y_train
    x_val, y_val = preprocessing.x_val, preprocessing.y_val
    print(preprocessing.get_vocab_size())
    print(preprocessing.get_vocab())
    print(x_train.shape)
    print(x_val.shape)

    embeddings=[]
    for text in data.texts: 
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states=outputs.last_hidden_state
        sentence_embeddings=torch.mean(torch.sum(last_hidden_states,dim=0),dim=0)
        embeddings.append(sentence_embeddings)
    embeddings=torch.stack(embeddings)

    #reduce the dimensionality to 5; size of the local neighborhood = 15
    umap_embeddings = umap.UMAP(n_neighbors=15, 
                                n_components=5, 
                                metric='cosine').fit_transform(embeddings.detach().numpy())

    cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

    docs_df = pd.DataFrame(data.texts, columns=["Document"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Document': ' '.join})
    print(docs_per_topic)

    tf_idf, count = c_tf_idf(docs_per_topic.Document.values, m=len(data.texts))

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df); print(topic_sizes.head(10))

    print(top_n_words[0][:])
    print(top_n_words[2][:])

    umap_data = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings.detach().numpy())
    hover_data = pd.DataFrame({'index':[i for i  in range(len(data.texts))],
                           'label':cluster.labels_,
                           'tweet':data.texts})

    p = umap.plot.interactive(umap_data, labels=cluster.labels_, hover_data=hover_data, point_size=10,color_key_cmap='Paired',background='black')
    output_file(filename="aa/results/clustering.mov", title="Static HTML file")
    save(p)
    """
    output_notebook()
    show(p)
    """
if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)
