
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from threading import Thread
import os, math


# # take all the trained data frames and make them big dicts by 
# 
# word-class: probability
# 

# In[2]:


def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    
def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

def saveFrame(df, name):
    df.to_csv(name+".csv", index=False, sep=",", header=True)
    save_it_all(df, name+".pkl")
    


# In[3]:


# load in raw data frame
## trained data info data set for use:
raw_trained_frame = load_objects("../RawFiles/raw_data_info_frame.pkl")


# In[4]:


raw_trained_frame.head()


# In[7]:


# get the dictionary for raw:
rawBigDict = {}
for docType in raw_trained_frame["Type"]:
    for key, value in raw_trained_frame["WordCount"][raw_trained_frame["Type"]==docType].tolist()[0].items():
        #print(key + str(value))
        rawBigDict[key+"-"+docType]=value["probability"]


# In[8]:


rawBigDict
save_it_all(rawBigDict, "../RawFiles/RawbigDict.pkl")

