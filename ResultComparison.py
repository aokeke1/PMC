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
            scores.append(np.sum(ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])))
        else:
            scores.append(L.ratio(personDict[p[0]][propToShow],personDict[p[1]][propToShow]))
    
    #Make Histogram
    plt.hist(scores)
    plt.title(tag)
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    
def makeDataForML(matches,skeptical):
    fileName = "C:/Users/arinz/Desktop/2016-2017/Projects/PatientMatchingChallenge/FinalDataset.csv"
    badEntries,sameThing = pkl.load(open('filters2.p','rb'))
    personDict,valueCounter,personDict2 = PM.loadData(fileName,badEntries,sameThing)
    
    #Collect positives
    positives = set()
    for p in matches:
        if (personDict['EnterpriseID'][p[0]][6]=="M" and personDict['EnterpriseID'][p[1]][6]=="F") or\
           (personDict['EnterpriseID'][p[0]][6]=="F" and personDict['EnterpriseID'][p[1]][6]=="M"):
               continue
        else:
            positives.add(p)
    #Collect negatives
    negatives = set()
    negatives.update(skeptical)
    for i in range(1,19):
        if tags[i] not in ['SSN','PHONE','FIRST','LAST','PHONE2']:
            continue

        dictConsidered = personDict[tags[i]]
        if len(negatives)>=(1+(0.9/0.1))*len(positives):
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
    
    #Perpare the samples
    ratioVec = np.vectorize(L.ratio)
    
    
    numTestVal = int(len(positives)/4)
    numTrain = len(positives) - 2*numTestVal
    
    testSet = np.zeros((numTestVal,18))
    valSet = np.zeros((numTestVal,18))
    trainSet = np.zeros((numTrain,18))
    
    #Test Set
    testLabels = []
    i = 0
    for i in range(numTestVal):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])
        testSet[i,:] = vec
        testLabels.append(1)
        i += 1
        for i in range(9):
            p = negatives.pop()
            info1 = personDict2['EnterpriseID'][p[0]]
            info2 = personDict2['EnterpriseID'][p[1]]
            vec = ratioVec(info1[1:],info2[1:])
            testSet[i,:] = vec
            testLabels.append(-1)
            i += 1
    #Validation Set
    valLabels = []
    i = 0
    for i in range(numTestVal):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])
        valSet[i,:] = vec
        valLabels.append(1)
        i += 1
        for i in range(9):
            p = negatives.pop()
            info1 = personDict2['EnterpriseID'][p[0]]
            info2 = personDict2['EnterpriseID'][p[1]]
            vec = ratioVec(info1[1:],info2[1:])
            valSet[i,:] = vec
            valLabels.append(-1)
            i += 1
    #Test Set
    trainLabels = []
    i = 0
    for i in range(numTestVal):
        p = positives.pop()
        info1 = personDict2['EnterpriseID'][p[0]]
        info2 = personDict2['EnterpriseID'][p[1]]
        vec = ratioVec(info1[1:],info2[1:])
        trainSet[i,:] = vec
        trainLabels.append(1)
        i += 1
        for i in range(9):
            p = negatives.pop()
            info1 = personDict2['EnterpriseID'][p[0]]
            info2 = personDict2['EnterpriseID'][p[1]]
            vec = ratioVec(info1[1:],info2[1:])
            trainSet[i,:] = vec
            trainLabels.append(-1)
            i += 1
    dataToSave = ((trainSet,trainLabels),(valSet,valLabels),(testSet,testLabels))
    output = open('MLdata.pkl', 'wb')
    pkl.dump(dataToSave,output)
    output.close()
if __name__=="__main__":
    personDict = loadPeople()
    
    matches,skeptical = loadMatches(8)
    
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