import pandas as pd
import pickle
from threading import Thread
import os

def main():
    print("main")
    # load files
<<<<<<< HEAD
=======
    # test
    # test=  {"class":4}
    dataTrained = load_objects("./StemStopFiles/dataInfoStopStemmingEdit.pkl")
    # results
    resultFrame = load_objects("./StemStopFiles/resultSSFrame.pkl")
    # vocab
    vocab = load_objects("./StemStopFiles/vocabStopStemmingEdit.pkl")
    #load test file
    testFrame = pd.read_csv("./StemStopFiles/testSSData.csv" ,sep=",", names=("Type","Document"))
>>>>>>> 8b01d0c1fb2a8e8039017c474522833efd1b216d

    ## trained data info data set for use:
    trainedFrame = load_objects("./RawFiles/raw_data_info_frame.pkl")

    ## vocabulary training set for use:
    vocabulary = load_objects("./RawFiles/raw_vocabulary.pkl")

    # test data
    testFrame = pd.read_csv("./RawFiles/raw_test_data.csv", sep=",")

<<<<<<< HEAD
    resultFrame = pd.read_csv("./RawFiles/raw_result_frame.csv", sep=",")
=======
    jobList = []
    for doc in testFrame["Document"]:
        indexVal = testFrame[testFrame["Document"] == doc].index.values.astype(int)[0]
        print(str(indexVal))
        thread = Thread(target = process, args = (dataTrained, doc, vocab, resultFrame, indexVal, ))
        jobList.append(thread)
>>>>>>> 8b01d0c1fb2a8e8039017c474522833efd1b216d

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

    countS = 0
    for job in jobList:
        print("Started: " + str(countS))
        countS = countS + 1
        job.start()

    countE = 0
    for job in jobList:
        print("Ended: " + str(countE))
        countE = countE + 1
        job.join()
    save_it_all(resultFrame, "./StemStopFiles/ResultFrameFinal.pkl")
    resultFrame.to_csv("./StemStopFiles/ResultFrameFinal.csv", index=False, sep=",", header=True)

def classDict(dataFrame):
    classificationDict = {}
    for docType in dataFrame["Type"]:
        classificationDict[docType] = None
    return classificationDict

def process(dataTrained, doc, vocab, resultFrame, indexVal):
    infoDict = classDict(dataTrained)
    for docType in dataTrained["Type"]:
        # print(docType)
        wordProbs = 1
        for word in doc.split():
            if word in vocab:
                #print("Word: "+word + ", Prob: " + str(dataTrained['wordCount'][dataTrained["Type"] == docType].tolist()[0][word]["probability"]))
                wordProbs = wordProbs * dataTrained['wordCount'][dataTrained["Type"] == docType].tolist()[0][word]["probability"]
        infoDict[docType] = wordProbs * dataTrained['Probablility'][dataTrained["Type"] == docType].tolist()[0]
    result = getMaxClass(infoDict)
    # resultFrame["Predicted"].iloc[indexVal]
    resultFrame.at[indexVal, 'Predicted'] = result

# classify the new documents:
# read in data frames
def save_it_all(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_objects(file):
    with open(file, 'rb') as input:
        return pickle.load(input)

def read(fileName):
    with open(fileName) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def seperator(docs):
    txt = []
    for line in range(0, len(docs)):
        words = docs[line].split()
        item = (",").join([words[0], (" ").join(words[1:])])
        txt.append(item)
    return txt

def write(data):
    with open('./test_data.csv', 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def getMaxClass(dictionary):
    result = {"class": None, "max": None, "values": None}
    maxVal = max(dictionary.values())
    for key, value in dictionary.items():
        if maxVal == value:
            result["class"] = key
            result["max"] = value
            result["values"] = dictionary
    return result

if __name__ == "__main__": main()
