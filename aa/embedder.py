from logging import getLogger
import io
import numpy as np
#import torch
import os
import numpy as np
import sys
import shutil
import gzip
from pathlib import Path
from urllib.request import urlopen
import logging
from logging import getLogger
from .embedder_utils import download_model,read_txt_embeddings
from .config_utils import EmbConfig
from .preprocessing_utils import update_emb_config
#from gensim.models.fasttext import FastText
#import gensim

logger = getLogger()
logger.setLevel(logging.DEBUG)

def load_pretrained_embeddings(params=None):
    """
    Reload pretrained embeddings from FastText.
    """
    #Language 
    #lang=params.lg
    lang="fr"
    #pretrained_dir= the directory which stores fasttext embeddings = pretrained_emb
    pretrained_dir="aa/pretrained_emb" 
    file_name="cc.%s.300.vec" % lang
    path=os.path.join(pretrained_dir,file_name)
    update_emb_config({"vector_size":300})
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)

    if not Path(path).exists():
        path=download_model(lang,if_exists="ignore")
    else: 
        logger.info(f"Loading {lang} embeddings .... ")
    logger.info(f"{lang} embeddings is saved at {path} ")
    return path 
    """
    if path.split("/")[-1].endswith('.bin'):
        pass
        #return load_bin_embeddings(path, params)
    else:
        return read_txt_embeddings(path, params)
    """
def custom_w2v_embeddings(corpus_file,emb_config):
    logger.info("Train w2v embedding ...")
    """
    sentences = gensim.models.word2vec.LineSentence(corpus_file)
    # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
    #https://radimrehurek.com/gensim/models/word2vec.html
    model = gensim.models.Word2Vec(sentences, **emb_config)
    word_vectors = model.wv
    pretrained_dir="aa/pretrained_emb" 
    file_name="word2vec.%s.wordvectors" % emb_config.vector_size

    path=os.path.join(pretrained_dir,file_name)

    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)

    if Path(path).exists():
        logger.info("File exists => file will be overwritten")
    word_vectors.save(path)

    return path
    """
def custom_ft_embeddings(corpus_file,emb_config):
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