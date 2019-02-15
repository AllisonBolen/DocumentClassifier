import pandas as pd
import pickle
from threading import Thread
import os

def main():
    print("main")
    # load files
    # test
    # test=  {"class":4}
    dataTrained = load_objects("./StemStopFiles/dataInfoStopStemmingEdit.pkl")
    # print(dataTrained.head())
    # print(dataTrained['Probablility'][dataTrained["Type"] == "atheism"].tolist()[0])
    # test["docType"] = 1 * dataTrained['Probablility'][dataTrained["Type"] == "atheism"].tolist()[0]
    # print(test)
    # Pretrained data
    resultFrame = load_objects("./StemStopFiles/resultSSFrame.pkl")
    # results
    vocab = load_objects("./StemStopFiles/vocabStopStemmingEdit.pkl")
    #load test file
    testFrame = pd.read_csv("./test_data.csv" ,sep=",", names=("Type","Document"))

    jobList = []
    for doc in testFrame["Document"]:
        indexVal = testFrame[testFrame["Document"] == doc].index.values.astype(int)[0]
        print(str(indexVal))
        thread = Thread(target = process, args = (dataTrained, doc, vocab, resultFrame, indexVal, ))
        jobList.append(thread)
        break

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
    print(type(classificationDict))
    return classificationDict

def process(dataTrained, doc, vocab, resultFrame, indexVal):
    infoDict = classDict(dataTrained)
    for docType in dataTrained["Type"]:
        print(docType)
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
