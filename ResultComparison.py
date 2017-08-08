# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:32:30 2017

@author: aokeke
"""
import time, itertools,csv,operator
import Levenshtein as L
import numpy as np
import pickle as pkl

from matplotlib import pyplot as plt
import PatientMatcher as PM
tags = ['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB',\
        'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
        'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']

def loadMatches(versNum):
    matched = {}
    skeptical = {}
    
    fileName1 = "OUTPUTS/IDMatchTablev"+str(versNum)+"_1.csv"
    fileName2 = "OUTPUTS/IDMatchTablev"+str(versNum)+"_3.csv"
    fileName3 = "OUTPUTS/IDMatchTablev"+str(versNum)+"_2.csv"

    #Load matches
    myFile = open(fileName1)
    myFile.readline()
    reader = csv.reader(myFile.read().split('\n'), delimiter=',')
    for line in reader:
        if len(line)>0:
            matched[(line[0],line[1])] = float(line[2])
        
    #load additional matches
    if versNum in range(7):    
        myFile = open(fileName2)
        myFile.readline()
        reader = csv.reader(myFile.read().split('\n'), delimiter=',')
        for line in reader:
            if len(line)>0:
                matched[(line[0],line[1])] = float(line[2])
    
    #Load skeptical matches
    myFile = open(fileName3)
    myFile.readline()
    reader = csv.reader(myFile.read().split('\n'), delimiter=',')
    for line in reader:
        if len(line)>0:
            skeptical[(line[0],line[1])] = float(line[2])
        
    return matched,skeptical

def loadPeople():
    personDict = {}
    
    fileNames = ["FinalDataset-1.csv","FinalDataset-2.csv"]
    for fileName in fileNames:
        #Load matches
        myFile = open(fileName)
        myFile.readline()
        reader = csv.reader(myFile.read().split('\n'), delimiter=',')
        for line in reader:
            if len(line)>0:
                personDict[line[0]] = line
        
    return personDict

def findExclusiveBetweenTwoVers(vers1,vers2):
    matches1,skeptical1 = loadMatches(vers1)
    matches2,skeptical2 = loadMatches(vers2)
    matches1k,matches2k,skeptical1k,skeptical2k = set(matches1),set(matches2),set(skeptical1),set(skeptical2)

    for i in list(matches1):
        if i in matches2k:
            matches1.pop(i)
    for i in list(matches2):
        if i in matches1k:
            matches2.pop(i)
    for i in list(skeptical1):
        if i in skeptical2k:
            skeptical1.pop(i)
    for i in list(skeptical2):
        if i in skeptical1k:
            skeptical2.pop(i)
        
    print (len(matches1),"matches unique to version",str(vers1))
    print (len(matches2),"matches unique to version",str(vers2))
    print (len(skeptical1),"skeptical matches unique to version",str(vers1))
    print (len(skeptical2),"skeptical matches unique to version",str(vers2))
    return (matches1,skeptical1),(matches2,skeptical2)

def showInfo(p,personDict,score):
    """
    Given a tuple pair of EnterpriseIDs, displays the info for both IDs
    """
    info1 = personDict[p[0]]
    info2 = personDict[p[1]]
    print ("Score:",score)
    print ("Person A:",info1)
    print ("Person B:",info2)
    print ("-----------------")

def graphSimilarities(matches,personDict,propToShow = -1):
    """
    Makes a histogram of property
    """
    #Get graph label
    if propToShow<=0 or propToShow>18:
        tag = "Overall Scores"
        ratioVec = np.vectorize(L.ratio)
    else:
        tag = tags[propToShow] + " Scores"
    
    #Get Data
    scores = []
    for p in matches:
        if propToShow<=0 or propToShow>18:
            scoreVector = ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])*(np.array(personDict[p[0]][1:])!='')
            scores.append(np.sum(scoreVector))
        else:
            score = L.ratio(personDict[p[0]][propToShow],personDict[p[1]][propToShow])*(personDict[p[0]][propToShow]!='')
            scores.append(score)
    
    #Make Histogram
    plt.hist(scores)
    plt.title(tag)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    
def makeDataForML(matches,skeptical):
    #fileName = "C:/Users/arinz/Desktop/2016-2017/Projects/PatientMatchingChallenge/FinalDataset.csv"
    fileName = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/FinalDataset-1.csv"
    badEntries,sameThing = pkl.load(open('filters2.p','rb'))
    print ("Loaing data",flush=True)
    personDict,valueCounter,personDict2 = PM.loadData(fileName,badEntries,sameThing)
    print ("Finished loading data",flush=True)
    #Collect positives
    positives = set()
    for p in matches:
        if personDict['EnterpriseID'][p[0]][6] != personDict['EnterpriseID'][p[1]][6]:
               continue
        else:
            positives.add(p)
            
            
    print ("Number of positives: ",len(positives),flush=True)
    #Collect negatives
    negatives = set()
    negatives.update(skeptical)
    for i in range(1,19):
        if tags[i] not in ['SSN','PHONE','FIRST','LAST','PHONE2']:
            continue

        dictConsidered = personDict[tags[i]]
        print ("Tag: ",tags[i],"\nNumber of Negatives: ",len(negatives),flush=True)
        if len(negatives)>=((1+(0.6/0.4))*len(positives)):
            break
        for duplicatedEntry in dictConsidered:
            if duplicatedEntry=="":
                #skip the empty entries
                continue
            pairs = itertools.combinations(dictConsidered[duplicatedEntry],2)
            for p in pairs:
                info1b = personDict2['EnterpriseID'][p[0]]
                info2b = personDict2['EnterpriseID'][p[1]]
                k = tuple(sorted(p))                
                if k not in matches and k not in skeptical:
                        score = PM.getScorePair(info1b,info2b)
                        if score<4:
                            negatives.add(k)
                            if len(negatives)>=((1+(0.6/0.4))*len(positives)):
                                break
    
    #Perpare the samples
    ratioVec = np.vectorize(L.ratio)
    
    reference = {'test':{},'train':{},'val':{}}
    
    numTestVal = int(len(positives)/4)
    numTrain = len(positives) - 2*numTestVal
    numTestVal2 = int(len(negatives)/4)
    numTrain2 = len(negatives) - 2*numTestVal2
    
    testSet = np.zeros((numTestVal+numTestVal2,18))
    valSet = np.zeros((numTestVal+numTestVal2,18))
    trainSet = np.zeros((numTrain+numTrain2,18))
    
    #Test Set
    print ("Making test set",flush=True)
    testLabels = []
    ind = 0
    for i in range(numTestVal):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        testSet[ind,:] = vec
        testLabels.append(1)
        reference['test'][ind] = [info1,info2]
        ind += 1
    for j in range(numTestVal+numTestVal2-len(testLabels)):
        p = negatives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        testSet[ind,:] = vec
        testLabels.append(-1)
        reference['test'][ind] = [info1,info2]
        ind += 1
    #Validation Set
    print ("Making validation set",flush=True)
    valLabels = []
    ind = 0
    for i in range(numTestVal):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        valSet[ind,:] = vec
        valLabels.append(1)
        reference['val'][ind] = [info1,info2]
        ind += 1
    for j in range(numTestVal+numTestVal2-len(valLabels)):
        p = negatives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        valSet[ind,:] = vec
        valLabels.append(-1)
        reference['val'][ind] = [info1,info2]
        ind += 1
    #Training Set
    print ("Making training set",flush=True)
    trainLabels = []
    ind = 0
    for i in range(numTrain):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        trainSet[ind,:] = vec
        trainLabels.append(1)
        reference['train'][ind] = [info1,info2]
        ind += 1
    for j in range(numTrain+numTrain2-len(trainLabels)):
        p = negatives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])*(np.array(info1[1:])!='')
        trainSet[ind,:] = vec
        trainLabels.append(-1)
        reference['train'][ind] = [info1,info2]
        ind += 1
        
    print (trainSet.shape,valSet.shape,testSet.shape)
    print (len(trainLabels),len(valLabels),len(testLabels))
    dataToSave = ((trainSet,trainLabels),(valSet,valLabels),(testSet,testLabels),reference)
    output = open('MLdata.pkl', 'wb')
    pkl.dump(dataToSave,output)
    output.close()
    
def showSortedByScore(matches,personDict,shouldReverse=False):
    newDict = {}
    
    for p in matches:
        newDict[p] = getScoreFromRatio(p,personDict)
    
    sorted_x = sorted(newDict.items(), key=operator.itemgetter(1),reverse=not shouldReverse)
    sorted_matches = []
    for i in sorted_x:
        sorted_matches.append(i[0])
    
    for p in sorted_matches:
        showInfo(p,personDict,newDict[p])
        
        pause = input("continue? (y/n)")
        if pause=="n":
            return
    
def getScoreFromRatio(p,personDict):
    ratioVec = np.vectorize(L.ratio)
    scoreVector = ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])*(np.array(personDict[p[0]][1:])!='')
    return np.sum(scoreVector)
def getWeightedScoreFromRatio(p,personDict,theta=np.ones((1,18)),theta_0=0):
    ratioVec = np.vectorize(L.ratio)
    scoreVector = ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])*(np.array(personDict[p[0]][1:])!='')
    return (np.dot(theta,scoreVector)+theta_0)[0]
if __name__=="__main__":
    pass
#    personDict = loadPeople()
#    
#    matches,skeptical = loadMatches(8)

    makeDataForML(matches,skeptical)
    
#    graphSimilarities(matches,personDict,propToShow=0)
#    graphSimilarities(skeptical,personDict,propToShow=0)
    
#    (matches1,skeptical1),(matches2,skeptical2) = findExclusiveBetweenTwoVers(7,8)

#    pause = ""
#    while pause not in ['y','n']:
#        pause = input("show next? (y/n) ")
#    if pause=="y":
#        for pair in matches1:
#            showInfo(pair,personDict,matches1[pair])
#
#    pause = ""
#    while pause not in ['y','n']:
#        pause = input("show next? (y/n) ")
#    if pause=="y":
#        for pair in matches2:
#            showInfo(pair,personDict,matches2[pair])
#            
#    pause = ""
#    while pause not in ['y','n']:
#        pause = input("show next? (y/n) ")
#    if pause=="y":
#        for pair in skeptical1:
#            showInfo(pair,personDict,skeptical1[pair])
#            
#    pause = ""
#    while pause not in ['y','n']:
#        pause = input("show next? (y/n) ")
#    if pause=="y":
#        for pair in skeptical2:
#            showInfo(pair,personDict,skeptical2[pair])