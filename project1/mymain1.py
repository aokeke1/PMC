# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 06:32:55 2017

@author: aokeke
"""


import time
import numpy as np
import random
import itertools
import pickle as pkl
import csv

np.set_printoptions(threshold=np.nan)
tags = ['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB',\
        'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
        'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']

def getInfoIntoDicts(fileName):
    print ("Using:",fileName)
    myFile = open(fileName)
    
    
    listOfDicts = {}
    for i in range(19):
        listOfDicts[tags[i]] = {}
#        listOfDicts.append({})
    
    myFile.readline() #pass the first line with the labels

    reader = csv.reader(myFile.read().split('\n'), delimiter=',')
    for line in reader:
        info = np.array(line)
        if len(info)!=19:
            print ("Info length:",len(info),"\ninfo:",info)
            continue
            
        for i in range(1,19):
            if i==14 and info[i]=="N.Y.":
                info[i] = "NY"
            if info[i].lower()=="unk" or info[i].lower() =="unknown":
                info[i] = ""
            if info[i] not in listOfDicts[tags[i]]:
                listOfDicts[tags[i]][info[i]] = [info[0]]
            else:
                listOfDicts[tags[i]][info[i]].append(info[0])
                
        if info[0] not in listOfDicts[tags[0]]:
            listOfDicts[tags[0]][info[0]] = np.array(info)
        else:
            print ("duplicate ID? ",info)
    myFile.close()
    return listOfDicts


def dataProcessing(listOfDicts):
    #Remove SSNs and phone numbers that are overly used
    maxRepeatsSSN = 3 #thresholds chosen from observing the data
    maxRepeatsPhone = 4
    IDDict = listOfDicts["EnterpriseID"]
    SSNDict = listOfDicts["SSN"]
    Phone1 = listOfDicts["PHONE"]
    
    print ("Before: SSN has",len(SSNDict),"and PHONE has",len(Phone1))
    print ("Dealing with SSN...")
    for s in list(SSNDict):
        if s=="":
            #skip blank entries
            continue
        if len(SSNDict[s])>maxRepeatsSSN:
            for i in SSNDict[s]:
                IDDict[i][7] = ""
                SSNDict[""].append(i)
            SSNDict.pop(s)
                 
    print ("Dealing with first phone numbers...")
    for p in list(Phone1):
        if p=="":
            #skip blank entries
            continue
        if len(Phone1[p])>maxRepeatsPhone:
            for i in Phone1[p]:
                IDDict[i][15] = ""
                Phone1[""].append(i)
            Phone1.pop(p)
    print ("After: SSN has",len(SSNDict),"and PHONE has",len(Phone1))
    return listOfDicts

def findMatches(listOfDicts,thresh):
    #Check for entries that have more than thresh things in common
    print ("Threshold set to",thresh)
    potentialPairs = {}
    theta=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    for i in range(1,19):
        if tags[i] not in ['LAST','FIRST','SSN']:
            print ("Skipping",tags[i],"...",flush=True)
            continue
        print ("Working on",tags[i],"...",flush=True)

        dictConsidered = listOfDicts[tags[i]]
        for duplicatedEntry in dictConsidered:
            if duplicatedEntry=="":
                #skip the empty entries
                continue
            pairs = itertools.combinations(dictConsidered[duplicatedEntry],2)
            for p in pairs:
                k = tuple(sorted((p[0],p[1])))
                if k not in potentialPairs:
                    info1 = listOfDicts['EnterpriseID'][p[0]]
                    info2 = listOfDicts['EnterpriseID'][p[1]]
                    score =  getScore(info1,info2,theta)
                    if score>=thresh:
#                        print (score,"\n",info1,"\n",info2)
#                        pause = input("pause")
#                        if pause=="0":
#                            return potentialPairs
                        potentialPairs[k] = score
        print (len(potentialPairs),"potential pairs so far",flush=True)
            
    return potentialPairs
def getScore(info1,info2,theta=np.ones((18,)),theta_0=0):
    try:
        score = np.dot(theta,(info1[1:]==info2[1:])&(info1[1:]!=""))+theta_0
    except ValueError:
        print (info1,info2)
        print (len(info1),len(info2))
        raise ValueError
    
    #maybe add factors for mixed names and fields
    return score

def inDepthCheck(potentialPairs,listOfDicts,lowerBound=0,upperBound=19):
    matches = {}
    notMatches = {}
    skipped = set()
    yn=""
    for pair in list(potentialPairs):
        if potentialPairs[pair]<=lowerBound:
            if (listOfDicts['EnterpriseID'][pair[0]][2]==listOfDicts['EnterpriseID'][pair[1]][2] and \
                   listOfDicts['EnterpriseID'][pair[0]][5]==listOfDicts['EnterpriseID'][pair[1]][5] and \
                   listOfDicts['EnterpriseID'][pair[0]][1]==listOfDicts['EnterpriseID'][pair[1]][1]) and \
                  (not (listOfDicts['EnterpriseID'][pair[0]][1]=="" or listOfDicts['EnterpriseID'][pair[0]][2]=="" or listOfDicts['EnterpriseID'][pair[0]][5]=="" )):
                if shouldPrintExtra:
                    print ("________________________________________________________________")
                    print ("Match Score:",potentialPairs[pair])
                    print ("Person A:",listOfDicts['EnterpriseID'][pair[0]])
                    print ("Person B:",listOfDicts['EnterpriseID'][pair[1]])
                    print("Dark Gray area accepted.")
                matches[pair] = potentialPairs[pair]
            else:
                notMatches[pair] = potentialPairs[pair]
        elif potentialPairs[pair]>=upperBound:
            if listOfDicts['EnterpriseID'][pair[0]][2]!=listOfDicts['EnterpriseID'][pair[1]][2] or \
               listOfDicts['EnterpriseID'][pair[0]][5]!=listOfDicts['EnterpriseID'][pair[1]][5] or \
               listOfDicts['EnterpriseID'][pair[0]][1]!=listOfDicts['EnterpriseID'][pair[1]][1]:
                skipped.add(pair)
            else:
                matches[pair] = potentialPairs[pair]
        else:
            if (listOfDicts['EnterpriseID'][pair[0]][2]==listOfDicts['EnterpriseID'][pair[1]][2] and \
                   listOfDicts['EnterpriseID'][pair[0]][5]==listOfDicts['EnterpriseID'][pair[1]][5] and \
                   listOfDicts['EnterpriseID'][pair[0]][1]==listOfDicts['EnterpriseID'][pair[1]][1]) and \
                  (not (listOfDicts['EnterpriseID'][pair[0]][1]=="" or listOfDicts['EnterpriseID'][pair[0]][2]=="" or listOfDicts['EnterpriseID'][pair[0]][5]=="" )):
                if shouldPrintExtra:
                    print ("________________________________________________________________")
                    print ("Match Score:",potentialPairs[pair])
                    print ("Person A:",listOfDicts['EnterpriseID'][pair[0]])
                    print ("Person B:",listOfDicts['EnterpriseID'][pair[1]])
                    print("Gray area accepted.")
                matches[pair] = potentialPairs[pair]
            else:
                skipped.add(pair)
    while len(skipped)>0:
        print ("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print ("Skipped",len(skipped),"pairs.")
        print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        if not shouldManual:
            return matches,notMatches,skipped
        for pair in list(skipped):
            print ("________________________________________________________________")
            print ("Match Score:",potentialPairs[pair])
            print ("Person A:",listOfDicts['EnterpriseID'][pair[0]])
            print ("Person B:",listOfDicts['EnterpriseID'][pair[1]])
            yn = input("is this a pair leave blank to skip or type 0 to quit? (y/n) ")
            if yn=="y":
                matches[pair] = potentialPairs[pair]
                skipped.remove(pair)
            elif yn=="n":
                notMatches[pair] = potentialPairs[pair]
                skipped.remove(pair)
            elif yn=="0":
                return matches,notMatches,skipped
            
    return matches,notMatches,skipped
     
def makeTrainValTest2(matching,nonmatching,listOfDicts,fileLabelNum):
    
    
    totalNumNonMatch = min(len(nonmatching),19*len(matched))
    nonmatchKeys = random.sample(list(nonmatching),totalNumNonMatch)
    nonmatching2 = {}
    for p in nonmatchKeys:
        nonmatching2[p]=nonmatching[p]
    #Matching Data
    numValTestMatch = int(len(matched)/4)
    numTrainMatch = len(matched)-2*numValTestMatch
    
    matchingKeys = list(matching)
    valMatch = random.sample(matchingKeys,numValTestMatch)
    matchingKeys2 = []
    for k in matchingKeys:
        if k not in valMatch:
            matchingKeys2.append(k)
    testMatch = random.sample(matchingKeys2,numValTestMatch)
    trainMatch = []
    for k in matchingKeys2:
        if k not in testMatch:
            trainMatch.append(k)
    
    #Non-matching Data
    numValTestNonmatch = int(len(nonmatching2)/4)
    numTrainNonmatch = len(nonmatching2)-2*numValTestNonmatch
    nonmatchingKeys = list(nonmatching2)
    valNonmatch = random.sample(nonmatchingKeys,numValTestNonmatch)
    nonmatchingKeys2 = []
    for k in nonmatchingKeys:
        if k not in valNonmatch:
            nonmatchingKeys2.append(k)
    testNonmatch = random.sample(nonmatchingKeys2,numValTestNonmatch)
    trainNonmatch = []
    for k in nonmatchingKeys2:
        if k not in testNonmatch:
            trainNonmatch.append(k)
    
    trainData = np.zeros((numTrainMatch + numTrainNonmatch,18))
    valData = np.zeros((numValTestMatch + numValTestNonmatch,18))
    testData = np.zeros((numValTestMatch + numValTestNonmatch,18))
    
    reference = {'train':{},'val':{},'test':{}}
    #Training data
    i = -1
    for p in trainMatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        trainData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['train'][len(reference['train'])] = mergedInfo
    for p in trainNonmatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        trainData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['train'][len(reference['train'])] = mergedInfo
    #Validation data
    i = -1
    for p in valMatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        valData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['val'][len(reference['val'])] = mergedInfo
    for p in valNonmatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        valData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['val'][len(reference['val'])] = mergedInfo
    #Test data
    i = -1
    for p in testMatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        testData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['test'][len(reference['test'])] = mergedInfo
    for p in testNonmatch:
        i+=1
        info1 = listOfDicts['EnterpriseID'][p[0]]
        info2 = listOfDicts['EnterpriseID'][p[1]]
        vec = (info1[1:]==info2[1:])&(info1[1:]!="")
        testData[i,:] = vec
        
        mergedInfo = np.vstack((info1,info2))
        reference['test'][len(reference['test'])] = mergedInfo
    
    
    #Labels
    trainLabels = np.array([1]*numTrainMatch + [-1]*numTrainNonmatch)
    valLabels = np.array([1]*numValTestMatch + [-1]*numValTestNonmatch)
    testLabels = np.array([1]*numValTestMatch + [-1]*numValTestNonmatch)
    ext = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/"
    if fileLabelNum==2:
        ext += "small/"
    elif fileLabelNum==3:
        ext +="smaller/"
    fileNames = ["trainData","trainLabels","valData","valLabels","testData","testLabels","reference"]
    pkl.dump(trainData,open(ext+fileNames[0]+str(fileLabelNum)+".p","wb"))
    pkl.dump(trainLabels,open(ext+fileNames[1]+str(fileLabelNum)+".p","wb"))
    pkl.dump(valData,open(ext+fileNames[2]+str(fileLabelNum)+".p","wb"))
    pkl.dump(valLabels,open(ext+fileNames[3]+str(fileLabelNum)+".p","wb"))
    pkl.dump(testData,open(ext+fileNames[4]+str(fileLabelNum)+".p","wb"))
    pkl.dump(testLabels,open(ext+fileNames[5]+str(fileLabelNum)+".p","wb"))
    pkl.dump(reference,open(ext+fileNames[6]+str(fileLabelNum)+".p","wb"))
    
#    saveFullMatchedUnmatched2(matching,nonmatching,nonmatching2,fileLabelNum)

def saveFullMatchedUnmatched2(matching,nonmatching,nonmatching2,fileLabelNum):
    matchStr = "ID1,ID2,BASIC_SCORE\n"
    nonmatchStr = "ID1,ID2,BASIC_SCORE\n"
    nonmatchStr2 = "ID1,ID2,BASIC_SCORE\n"
    
    for p in matching:
        matchStr += p[0]+","+p[1]+","+str(matching[p])+"\n"
    for p in nonmatching:
        nonmatchStr += p[0]+","+p[1]+","+str(nonmatching[p])+"\n"
    for p in nonmatching2:
        nonmatchStr2 += p[0]+","+p[1]+","+str(nonmatching2[p])+"\n"
        
    ext = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/"
    if fileLabelNum==2:
        ext += "small/"
    elif fileLabelNum==3:
        ext +="smaller/"
    
    matchFileName = ext+"matching_pairs_"+str(fileLabelNum)+".csv"
    nonmatchFileName = ext+"nonmatching_pairs_"+str(fileLabelNum)+".csv"
    nonmatchFileName2 = ext+"nonmatching_pairs2_"+str(fileLabelNum)+".csv"
    
    mFile = open(matchFileName,"w")
    nmFile = open(nonmatchFileName,"w")
    nmFile2 = open(nonmatchFileName2,"w")
    
    mFile.write(matchStr)
    nmFile.write(nonmatchStr)
    nmFile2.write(nonmatchStr2)
    
    mFile.close()
    nmFile.close()    
    nmFile2.close() 
    
    
if __name__=="__main__":
    shouldManual = True
    shouldPrintExtra = False
    fileLabelNum = 3
    
    fileName1 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/FinalDataset-1.csv"
    fileName2 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/small/FinalDataset-1small.csv"
    fileName3 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/smaller/FinalDataset-1smaller.csv"
    fileList = [fileName1,fileName2,fileName3]
    startTime = time.time()
    listOfDicts = getInfoIntoDicts(fileList[fileLabelNum-1])
    endTime = time.time()
    print ("Loading data took ",(endTime-startTime)," seconds")
    listOfDicts = dataProcessing(listOfDicts)
     
    potentialPairs = findMatches(listOfDicts,thresh=3)
    print ("Number of pairs pre filtering: ",len(potentialPairs))
    
    matched,notMatches,skipped = inDepthCheck(potentialPairs,listOfDicts,lowerBound=6,upperBound=8)
    
    print ("Number of matching pairs: ",len(matched))
    print ("Number of non-matching pairs: ",len(notMatches))
    print ("Number of skipped pairs: ",len(skipped))
#    if len(skipped)==0:
    makeTrainValTest2(matched,notMatches,listOfDicts,fileLabelNum)
    print("saved")
    pass