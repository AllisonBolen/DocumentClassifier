import pandas as pd
import pickle, os, math
from threading import Thread
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

def main():
    print("main")
    # load files

    ## trained data info data set for use:
    # load raw data dict and vocab and raw frame
    raw_dict = load_objects("./RawFiles/rawBigDict.pkl")
    raw_trained_frame = load_objects("./RawFiles/raw_data_info_frame.pkl")
    raw_vocab = load_objects("./RawFiles/raw_vocabulary.pkl")

    # test data
    testFrame = pd.read_csv("./RawFiles/raw_test_data.csv", sep=",")

    right, total = process(raw_trained_frame, testFrame, raw_dict, raw_vocab)

    print("THE FINAL RESULT IS:")
    print(str(right/total) + "% predicated correctly!!")

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

def dictHelper(df):
    types = df["Type"]
    res = {}
    for ty in types:
        res[ty] = 0
    return res

def process(trained_frame, testFrame, chosenDict, vocab):
    # Classify this shit:
    rawResult = dictHelper(trained_frame)
    # Classify this shit:
    i = 0
    q = 0
    for index, row in testFrame.iterrows():
        print(i)
        i = i + 1
        classification = {}
        for docType in trained_frame["Type"]:
            probabilities = []
            doc = row["Document"]
            for word in doc.split():
                if word+"-"+docType in chosenDict:
                    probabilities.append(math.log(chosenDict[word+"-"+docType]))
                else:
                    probabilities.append(math.log(1/(trained_frame["WordPositions"][trained_frame["Type"]==docType]+len(vocab))))
            classification[docType] = math.log(trained_frame["ClassProbability"][trained_frame["Type"]==docType]) + sum(probabilities)

        maxClass = None
        for key, value in classification.items():
            if value == max(classification.values()):
                maxClass = key

        if maxClass == row["Type"]:
            print("Success! predicted: "+ maxClass + " real: "+row["Type"])
            rawResult[row["Type"]] = rawResult[row["Type"]] + 1
            q = q + 1


    for key, value in rawResult.items():
        print(key + ": " + str((value/sizes[key])*100))
    return q, i


if __name__ == "__main__": main()
