import pandas as pd 
label_dic = {}
labels = []
texts = []
num_classes = 0
"""
with open("aa/data/Campagne2017", "r", encoding="utf-8") as f: 
    for text in f.readlines():
        label = text.split("__"+ " ")[0].replace("__", "")
        text = text.replace("__" + label + "__" + " ", "")
        if label not in label_dic.keys():
            label_dic[label] = num_classes
            num_classes += 1
        label_int = label_dic[label]
        #labels += [label_int]
        labels += [label]
        texts += [text]

df= pd.DataFrame(list(zip(labels,texts)))
df.to_csv("aa/data/Campagne2017.csv",sep="\t", header=False, index=False)
"""
with open("aa/data/Campagne2017.test", "r", encoding="utf-8") as f: 
    texts=[text.strip('\n') for text in f.readlines()]
    labels=["Macron"]*len(texts)

df= pd.DataFrame(list(zip(labels,texts)))
df.to_csv("aa/data/Campagne2017test.csv",sep="\t", header=False, index=False)