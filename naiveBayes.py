import pandas as pd
import pickle
from threading import Thread
import os

def main():
    print("main")
    # load files

    ## trained data info data set for use:
    trainedFrame = load_objects("./RawFiles/raw_data_info_frame.pkl")

    ## vocabulary training set for use:
    vocabulary = load_objects("./RawFiles/raw_vocabulary.pkl")

    # test data
    testFrame = pd.read_csv("./RawFiles/raw_test_data.csv", sep=",")

    resultFrame = pd.read_csv("./RawFiles/raw_result_frame.csv", sep=",")

    # start predicting:
    testFrame["Predicted"] = None
    jobs = []
    for index, row in testFrame.iterrows():
        unique_id = index
        print(unique_id)
        classProb = classDict(trainedFrame)

        thread = Thread(target = process, args = (trainedFrame, resultFrame, row, index, classProb, vocabulary, ))
        jobs.append(thread)
        
    # start the model threads
    countS = 0
    for job in jobs:
        print("Started: " + str(countS))
        countS = countS + 1
        job.start()
    # wait for all threads to finish
    countE = 0
    for job in jobs:
        print("Ended:  " + str(countE))
        countE = countE + 1
        job.join()

    saveFrame(testFrame.copy(), "./RawFiles/ResultFrameFinal")

def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

def vocabDict(vocab, doc, n): # handles empty word sets
    nk = 0
    # n = the number of word postions for this document type
    probability = math.log(nk + 1) - math.log(n + len(vocab))

    newDict = {"count": 0, "probability":probability}
    vdict = dict((el,newDict) for el in vocab)
    return vdict

def saveFrame(df, name):
    df.to_csv(name+".csv", index=False, sep=",", header=True)
    save_it_all(df, name+".pkl")

def getMaxClass(dictionary):
    result = {"class": None, "max": None, "values": None}
    maxVal = max(dictionary.values())
    for key, value in dictionary.items():
        if maxVal == value:
            result["class"] = key
            result["max"] = value
            result["values"] = dictionary
    return result

# niave bayes caclulation
def classDict(df):
    classificationDict = {}
    for docType in df["Type"]:
        classificationDict[docType] = None
    return classificationDict

def process(trainedFrame, resultFrame, row, index, classProb, vocab):
    for docType in trainedFrame["Type"]:
        #print(docType)
        wordProbs = 1
        for word in row["Document"].split():
            if word in vocab:
                i = 0
                #print("Word: "+word + ", Prob: " + str(dataTrained['wordCount'][dataTrained["Type"] == docType].tolist()[0][word]["probability"]))
                wordProbs = wordProbs * trainedFrame['WordCount'][trainedFrame["Type"] == docType].tolist()[0][word]["probability"]
        classProb[docType] = wordProbs * trainedFrame['ClassProbability'][trainedFrame["Type"] == docType].tolist()[0]
    result = getMaxClass(classProb)
    # # resultFrame["Predicted"].iloc[index]
    print(result)

    row['Predicted'] = result

if __name__ == "__main__": main()
