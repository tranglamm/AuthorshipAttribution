import json 

def load_json(file_json):
    with open(file_json,"r",encoding="utf-8") as f: 
        data = json.load(f)
    return data
    
def update_model_config(dict_config):
    with open("aa/ressources/config/model_config.json", "r+") as file:
        data = json.load(file)
        #if list(dict_config.keys())[0] not in data:
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)

def update_emb_config(dict_config):
    with open("aa/ressources/config/emb_config.json", "r+") as file:
        data = json.load(file)
        #if list(dict_config.keys())[0] not in data:
        data.update(dict_config)
        file.seek(0)
        json.dump(data, file,indent=4)
