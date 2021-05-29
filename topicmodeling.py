from aa import *
import pandas as pd 
import argparse
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.feature_extraction.text import CountVectorizer



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
    data=PrepareData.for_training(params)
    topModel=TopicModeling.from_transformers(data, params.transformer_model)
    topic_sizes = topModel.get_topic_sizes()
    print(topic_sizes)
    if params.get_top_words: 
        """
        Get words per topic
        """
        id_topic = params.get_top_words
        words_topic = topModel.get_top_n_words()[id_topic][:]
        df_top_words =  pd.DataFrame(words_topic, columns =['Word', 'TF_IDF'])
        print(f"Top n words of {id_topic} is:\n{df_top_words}")
    
    if params.get_doc:
        """
        Get doc per topic
        """
        id_topic = params.get_doc
        docs_df = topModel.create_docs_df()
        docs_topic = docs_df[docs_df['Topic']==id_topic]
        print(docs_topic)
    
    if params.visualize:
        topModel.visualize()

        
if __name__=="__main__":
    parser=get_parser()
    params = parser.parse_args()
    main(params)

