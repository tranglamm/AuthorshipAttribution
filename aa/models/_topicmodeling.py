import umap
import hdbscan
from umap import plot
from bokeh.plotting import show, save, output_notebook, output_file
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
import pandas as pd
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



class TopicModeling: 
    def __int__(self,data,embeddings):
        """
        Here you can pass your own embeddings 
        """
        self.data = data
        self.embeddings = embeddings

    @classmethod
    def from_transformers(cls, data, transformer_model):
        """
        Visit to HuggingFace's to get the list of pretrained models
        """
        tokenizer= AutoTokenizer.from_pretrained(transformer_model)
        model = AutoModel.from_pretrained(transformer_model)
        embeddings=[]
        for text in data.texts: 
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states=outputs.last_hidden_state
            sentence_embeddings=torch.mean(torch.sum(last_hidden_states,dim=0),dim=0)
            embeddings.append(sentence_embeddings)
        embeddings=torch.stack(embeddings)
        return cls(data,embeddings)
    
    def clustering(self):
        """
        Documents with similar topics are clustered together => find the topics within theses clusters
        """
        umap_embeddings = umap.UMAP(n_neighbors=15, 
                                n_components=5, 
                                metric='cosine').fit_transform(self.embeddings.detach().numpy())

        cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                            metric='euclidean',                      
                            cluster_selection_method='eom').fit(umap_embeddings)
        return cluster 

    def create_docs_df(self):
        cluster = self.clustering()
        texts = self.data.texts
        docs_df = pd.DataFrame(texts, columns=["Document"])
        docs_df['Topic'] = cluster.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        return docs_df 
    
    def get_docs_per_topic(self):
        docs_df = self.create_docs_df()
        docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Document': ' '.join})
        return docs_per_topic

    def get_top_n_words(self):
        docs_per_topic = self.get_docs_per_topic()
        tf_idf, count = c_tf_idf(docs_per_topic.Document.values, m=len(self.texts))
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
        return top_n_words
    
    def get_topic_sizes(self):
        docs_df = self.create_docs_df()
        topic_sizes = extract_topic_sizes(docs_df)
        return topic_sizes
    
    def visualize(self):
        cluster = self.clustering()
        umap_data = umap.UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric='cosine').fit(self.embeddings.detach().numpy())
        hover_data = pd.DataFrame({'index':[i for i  in range(len(self.data.texts))],
                            'topic':cluster.labels_,
                            'label':self.data.labels,
                            'tweet':self.data.texts})

        p = umap.plot.interactive(umap_data, labels=cluster.labels_, hover_data=hover_data, point_size=10,color_key_cmap='Paired',background='black')
        output_file(filename="aa/results/clustering.html", title="Clustering HTML file")
        save(p)


