"""
Allison Bolen
CIS 678
WIN19
"""
# get imports # import files
from threading import Thread
import queue as que
import pandas as pd
import os, pickle
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from collections import Counter

stemmer = SnowballStemmer("english")
cachedStopWords = stopwords.words("english")

# saves objects
def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# loads objects
def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

# merge dictionarys
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

# seperate out the catagory from teh document
# on initally reading the training data file
def seperator(docs):
    print(len(docs))
    txt = []
    typesInfo = []
    docsInfo = []
    for line in docs:
        words = line.split()
        content = (" ").join(words[1:])
        item = (", ").join([words[0], content])
        txt.append(item)
        typesInfo.append(words[0])
        docsInfo.append(content)
    return txt, typesInfo, docsInfo

# Write out the initial csv file
def write(data):
    print(type(data))
    with open('./training_data.csv', 'w') as f:
        for item in data:
            f.write("%s\n" % item)

# read the forumTraining data
def read(fileName):
    with open(fileName) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

# makes an intial dictionary wehre all probabilities
# for every word is as though the word is of count zero
def vocabDict(vocab, n): # handles empty word sets
    nk = 0
    # n = the number of word postions for this document type
    probability = ((nk + 1) / (n + len(vocab)))

    newDict = {"count": 0, "probability":probability}
    vdict = dict((el,newDict) for el in vocab)
    return vdict

# gets the word occurance in a document and tracks its probability, merges with
# the empty vocab dict to save time.
def wordCount(doc, vocab, n): # get actual word count for each word and create dict frame to track the info
    counts = dict(Counter(doc.tolist()[0].split()))
    for key, value in counts.items():
        nk = value
        probability = ((nk + 1) / (n + len(vocab)))
        newDict = {"count": nk, "probability": probability}
        counts[key] = newDict
    emptyVocab = vocabDict(vocab, n)
    dicts = [merge_two_dicts(emptyVocab, counts)]
    return dicts

# saves thing as a csv and a pickled object
def saveFrame(df, name):
    df.to_csv(name+".csv", index=False, sep=",", header=True)
    save_it_all(df, name+".pkl")

# In[101]:
# # load in files
# # read file line by line
fileName = "../forumTraining.data"
data = read(fileName)
len(data)
sep_data, typesInfo, docsInfo = seperator(data)
# write(sep_data) not needed the raw csv it makes IS BROKEN
# print(len(list(set(typesInfo))))
# print(list(set(typesInfo)))
dataFrameRaw = pd.DataFrame({"Type": typesInfo, "Document": docsInfo})
saveFrame(dataFrameRaw, "raw_training_data.csv")

# In[102]:
raw_training_data = pd.read_csv("./raw_training_data.csv", sep=",")
raw_training_data.head()

# ### Trianing data raw is held in the dataframe "raw_training_data"

# In[103]:
# create the raw vocabulary set:
raw_vocabulary = " ".join(raw_training_data["Document"].tolist()) # concatinates all documents in the data frame
raw_vocabulary = list(set(raw_vocabulary.split()))
print(len(raw_vocabulary))
save_it_all(raw_vocabulary, "./raw_vocabulary.pkl")

# ### The raw vocabulary is saved in the above cell and refrenced as "raw_vocabulary"

# # This is the learning step: make it dynamic!
# For each class cj (document type) in C
#
#     1. Docsj ← training documents for which the classification is cj
#
#     2. Probability estimate of a particular class: P(cj) = |Docsj| / |training documents|
#
#     3. Textj ← create a single document per class (concatenate all Docsj)
#
#     4. n = total number of word positions in Textj
#
#     5. For each word wk in Vocabulary nk = number of times wk occurs in Textj
#
#     6. Estimate of word occurrence for particular document type: P(wk | cj) = (nk + 1) / (n + |Vocabulary|)
#
# Probability of kth word in vocabulary, given a document of type j

# In[104]:
# load in docs you want to use:
## trainign data set for use:
trainFrame = pd.read_csv("./raw_training_data.csv", sep=",")
## vocabulary training set for use:
vocabulary = load_objects("./raw_vocabulary.pkl")
# validate:
print(type(vocabulary))
trainFrame.head()

# In[105]:
# single docs per class
types = list(set(trainFrame["Type"]))
singleFrame = pd.DataFrame({"Type":types , "Document": None})
# singleFrame.head(20)
for docType in singleFrame["Type"]:
    #print(trainFrame["Document"][trainFrame["Type"]==docType])
    singleFrame["Document"][singleFrame["Type"]== docType] = " ".join(trainFrame["Document"][trainFrame["Type"]==docType].tolist())
singleFrame.head(30)

# In[106]:
# make a copy of the single frame
rawDataInfoFrame = singleFrame.copy()

# In[107]:
# probability estimate of the class
rawDataInfoFrame["ClassProbability"] = None
for docType in rawDataInfoFrame["Type"]:
    rawDataInfoFrame["ClassProbability"][rawDataInfoFrame["Type"]== docType] = len(trainFrame["Document"][trainFrame["Type"]== docType].tolist()) / trainFrame["Type"].count()

# In[109]:
# get the word postions:
rawDataInfoFrame["WordPositions"] = None
for docType in rawDataInfoFrame["Type"]:
    rawDataInfoFrame["WordPositions"][rawDataInfoFrame["Type"]== docType] = len(rawDataInfoFrame["Document"][rawDataInfoFrame["Type"]== docType].tolist()[0].split())

# In[111]:
# word count
rawDataInfoFrame["WordCount"] = None
for docType in rawDataInfoFrame["Type"]:
    n = rawDataInfoFrame["WordPositions"][rawDataInfoFrame["Type"] == docType].tolist()[0]
    print(n)
    rawDataInfoFrame["WordCount"][rawDataInfoFrame["Type"] == docType] = wordCount(rawDataInfoFrame["Document"][rawDataInfoFrame["Type"]==docType], vocabulary, n)
