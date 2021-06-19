import json 
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import torch 


class Config: 
    def __init__(self,**kwargs):
        self.emb_dim = kwargs.pop("emb_dim", 128)
        self.max_length = kwargs.pop("max_length", 50)
        self.vocab_size = kwargs.pop("vocab_size", None)
        self.nb_filters = kwargs.pop("nb_filters", 512)
        self.num_classes = kwargs.pop("num_classes", None)
        self.filter_sizes = kwargs.pop("nb_filters", [3])
        self.dense_layer_size = kwargs.pop("dense_layer_size", 100)
        self.lr = kwargs.pop("lr", 1e-3)
        self.dropout_val = kwargs.pop("dropout_val",0.2)
        self.num_epochs = kwargs.pop("num_epochs",20)
        self.batch_size = kwargs.pop("batch_size",800)


        #Model Attention configuration 
        self.rnn_dim=kwargs.get("rnn_dim",100)
        

        #Embedding configuration 
        self.vector_size = kwargs.pop("vector_size", 128)
        self.window = kwargs.pop("window",10)
        self.workers = kwargs.pop("workers",8)
        self.sg = kwargs.pop("sg",1)
        self.hs = kwargs.pop("hs",0)
        self.cbow_mean = kwargs.pop("cbow_mean",1)
        self.ns_exponent=kwargs.pop("ns_exponent",0.75)
        self.alpha = kwargs.pop("alpha", 0.025)
        self.min_count = kwargs.pop("min_count", 0)
        self.sample = kwargs.pop("sample", 0.001)
        self.negative = kwargs.pop("negative", 5)
        self.epochs = kwargs.pop("epochs",5)
        self.sorted_vocab = kwargs.pop("sorted_vocab",1)
        self.max_vocab_size = kwargs.pop("max_vocab_size",None)
        self.threads = kwargs.pop("threads",12)
        self.min_n = kwargs.pop("min_n",3)
        self.max_n = kwargs.pop("max_n",6)
        self.bucket = kwargs.pop("bucket",2000000)


        #Transformer optimizer loss 
        self.loss = kwargs.pop("loss", "CrossEntropy")
        self.optimizer = kwargs.pop("optimizer", "Adam")

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "Config":
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)


    
class ModelConfig(Config): 
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    
class EmbConfig(Config): 
    """
    For more information, please feel free to take a look at 
    https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
    """
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
        


class TrainingConfig(Config): 
    def __init__(self,
                **kwargs):
        super().__init__(**kwargs)
