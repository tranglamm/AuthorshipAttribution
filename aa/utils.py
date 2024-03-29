import json 
from .file_utils import CONFIG_JSON

transformer_config = CONFIG_JSON["transformer_config"]
model_config = CONFIG_JSON["model_config"]
emb_config= CONFIG_JSON["emb_config"]

def load_json(file_json):
    with open(file_json,"r",encoding="utf-8") as f: 
        data = json.load(f)
    return data
"""
def update_transformer_config(dict_config):
    with open(transformer_config, "r+") as file:
        data = json.load(file)
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)

def update_emb_config(dict_config):
    with open(emb_config, "r+") as file:
        data = json.load(file)
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)

def update_model_config(dict_config):
    with open(model_config, "r+") as file:
        data = json.load(file)
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)
"""

def update_transformer_config(**kwargs):
    with open(transformer_config, "r+") as file:
        data = json.load(file)
        for k,v in kwargs.items():
            data[k]=v
        file.seek(0)
        json.dump(data, file,indent=4)

def update_emb_config(**kwargs):
    with open(emb_config, "r+") as file:
        data = json.load(file)
        for k,v in kwargs.items():
            data[k]=v
        file.seek(0)
        json.dump(data, file,indent=4)

def update_model_config(**kwargs):
    with open(model_config, "r+") as file:
        data = json.load(file)
        for k,v in kwargs.items():
            data[k]=v
        file.seek(0)
        json.dump(data, file,indent=4)
        


