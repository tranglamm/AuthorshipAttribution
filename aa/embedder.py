"""
Load Embedding for Layer Embedding of CNN model 

@author: Trang Lam - github.com/tranglamm
"""

from logging import getLogger
import io
import numpy as np
import os
import numpy as np
import sys
import shutil
import gzip
from pathlib import Path
from urllib.request import urlopen
import logging
from logging import getLogger

import fasttext.util
from .embedder_utils import download_model,read_txt_embeddings
from .config_utils import EmbConfig
from .preprocessing_utils import *
from gensim.models.fasttext import FastText
import gensim

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)
def load_pretrained_embeddings(params=None):
    """
    Reload pretrained embeddings from FastText.
    """
    logger.info(
      "Due to the dimension of pretrained embeddings FastText is set to 300.\
      => The emb_dim of your model will also set to 300. If you want to reduce dimension, please add --reduce_dim (It can take a lot of time) "
    )
    update_emb_config({"vector_size":300})
    update_model_config({"emb_dim":300})
    lang=params.lg

    #pretrained_dir= the directory which stores fasttext embeddings = pretrained_emb
    pretrained_dir="aa/ressources/pretrained_emb" 
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    
    
    format_vec = 'bin' if params.reduce_dim != 300 else 'vec'
    file_name="cc.%s.300.%s" % (lang,format_vec)
    path=os.path.join(pretrained_dir,file_name)
    
    if not Path(path).exists():
        path=download_model(lang,format_vec = format_vec, if_exists="ignore")
    else: 
        logging.info(f"Loading {lang} embeddings .... ")

    #If --reduce_dim is called => update vector_size + emb_dim for model_config.json and emb_config.json
    if params.reduce_dim != 300: 
        update_emb_config({"vector_size":params.reduce_dim})
        update_model_config({"emb_dim":params.reduce_dim})
        ft = fasttext.load_model(path)
        fasttext.util.reduce_model(ft, params.reduce_dim)
        file_name="cc.%s.%s.%s" % (lang,params.reduce_dim,format_vec)
        path=os.path.join(pretrained_dir,file_name)
        ft.save_model(path)
    logging.info(f"{lang} embeddings is saved at {path} ")
    return path 
    

def custom_w2v_embeddings(texts,emb_config):
    logging.info("Train w2v embedding ...")
    
    #sentences = gensim.models.word2vec.LineSentence(corpus_file)
    # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    #https://radimrehurek.com/gensim/models/word2vec.html
    texts=[text.split() for text in texts]
    model = gensim.models.Word2Vec(texts, 
    size=emb_config.vector_size,
    window=emb_config.window,
    workers=emb_config.workers,
    sg=emb_config.sg)
    word_vectors = model.wv
    pretrained_dir="aa/ressources/pretrained_emb" 
    file_name="word2vec_%s.wordvectors" % emb_config.vector_size

    path=os.path.join(pretrained_dir,file_name)

    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)

    if Path(path).exists():
        logging.info("File exists => file will be overwritten")
    word_vectors.save(path)

    return path
    
def custom_ft_embeddings(texts,emb_config):
    pass
    """
    model = FastText(vector_size=100, **emb_config)

    # build the vocabulary
    model.build_vocab(corpus_file=corpus_file)
    # train the model
    model.train(
        corpus_file=corpus_file, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words,
    )

    model.save(tmp.name, separately=[])

    # Load back the same model.
    loaded_model = FastText.load(tmp.name)
    print(loaded_model)
    """