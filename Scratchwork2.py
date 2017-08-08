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

ratioVec = np.vectorize(L.ratio)
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

def loadPeople2():
    personDict = {}
    MRNGroups = {}
    fileNames = ["FinalDataset-1.csv","FinalDataset-2.csv"]
    for fileName in fileNames:
        #Load matches
        myFile = open(fileName)
        myFile.readline()
        reader = csv.reader(myFile.read().split('\n'), delimiter=',')
        for line in reader:
            if len(line)>0:
                personDict[line[0]] = line
                MRN = line[12]
                if MRN!="":
                    if len(MRN) not in [6,7]:
                        print ("MRN: ",MRN)
                    else:
                        if len(MRN)==7:
                            MRN = MRN[0:4]+"xxx"
                        elif len(MRN)==6:
                            MRN = MRN[0:4]+"xx"
                        if MRN in MRNGroups:
                            MRNGroups[MRN].add(line[0])
                        else:
                            MRNGroups[MRN] = {line[0]}
                            
        
    return personDict,MRNGroups

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
    
    scoreVector = ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])*(np.array(personDict[p[0]][1:])!='')
    return np.sum(scoreVector)
def getWeightedScoreFromRatio(p,personDict,theta=np.ones((1,18)),theta_0=0):

    scoreVector = ratioVec(personDict[p[0]][1:],personDict[p[1]][1:])*(np.array(personDict[p[0]][1:])!='')
    return (np.dot(theta,scoreVector)+theta_0)[0]

def lookForMatchesFromMRN():
    cutoff = 5
    highCutoff = 11
    lowCutoffa = 8
    lowCutoffb = 1.5
    genderCutoff = 2.5
    personDict,MRNDict = loadPeople2()
    matches,skeptical = loadMatches(8)
    newMatches = {}
    count = 0
    for MRN in MRNDict:
        count += 1
        if count%int(len(MRNDict)/25)==0:
            print (round(100*count/len(MRNDict),3),"% complete")
            print (len(newMatches),"new matches found so far",flush=True)
        consideredPeople = MRNDict[MRN]
        pairs = itertools.combinations(consideredPeople,2)
        for p in pairs:
            k = tuple(sorted(list(p)))
            if (k not in matches) and (k not in skeptical) and (k not in newMatches):
                score = getScoreFromRatio(k,personDict)
                info1 = personDict[k[0]]
                info2 = personDict[k[1]]
                if (score>=cutoff and ((info1[1]==info2[1]) or (info1[2]==info2[2]) or (info1[5]==info2[5]))) or score>highCutoff:
                    #SSN too different
                    if info1[7]!='' and info2[7]!='' and L.distance(info1[7],info2[7])>2:
                        continue
                    #Birthday too different
                    if info1[5]!='' and info2[5]!='' and L.distance(info1[5],info2[5])>1:
                        continue
                    #Differing genders
                    if ((info1[4]=="M" and info2[4]=="F")or(info1[4]=="F" and info2[4]=="M")) and sum(ratioVec(info1[1:4],info2[1:4])*(np.array(info1[1:4])!=''))<genderCutoff:
                        continue
                    if score<lowCutoffa and sum(ratioVec(info1[1:4],info2[1:4])*(np.array(info1[1:4])!=''))<lowCutoffb:
                        continue
                    newMatches[k] = score
                    
    print (len(newMatches),"new matches found")
    
    fileName1 = "MRNIDMatchesCutoff"+str(cutoff)+".csv"
    fileName2 = "MRNFullInfoMatches"+str(cutoff)+".csv"
    saveMatches(newMatches,personDict,fileName1,fileName2)
    return newMatches

def saveMatches(matches,personDict,fileOutName1,fileOutName2):
    """
    Save matches to a file
    """
    sorted_x = sorted(matches.items(), key=operator.itemgetter(1),reverse=True)
    sorted_matches = []
    for i in sorted_x:
        sorted_matches.append(i[0])
    with open(fileOutName1, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['EnterpriseID1','EnterpriseID2','MATCH_SCORE'])
        for p in sorted_matches:
            spamwriter.writerow([p[0],p[1],str(matches[p])])
            
    with open(fileOutName2, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB','GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME','MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS'])
        for p in sorted_matches:
            spamwriter.writerow(list(personDict[p[0]]))
            spamwriter.writerow(list(personDict[p[1]]))
            spamwriter.writerow([])
            
if __name__=="__main__":
    pass
#    personDict = loadPeople()
#    
#    matches,skeptical = loadMatches(8)

#    makeDataForML(matches,skeptical)
    
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
    newMatches = lookForMatchesFromMRN()