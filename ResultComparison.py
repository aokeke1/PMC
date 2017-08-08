# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:32:30 2017

@author: aokeke
"""
import time, itertools,csv,operator

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
    
    fileName = "FinalDataset-1.csv"
    
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

if __name__=="__main__":
#    personDict = loadPeople()
    (matches1,skeptical1),(matches2,skeptical2) = findExclusiveBetweenTwoVers(7,8)

    pause = ""
    while pause not in ['y','n']:
        pause = input("show next? (y/n) ")
    if pause=="y":
        for pair in matches1:
            showInfo(pair,personDict,matches1[pair])

    pause = ""
    while pause not in ['y','n']:
        pause = input("show next? (y/n) ")
    if pause=="y":
        for pair in matches2:
            showInfo(pair,personDict,matches2[pair])
            
    pause = ""
    while pause not in ['y','n']:
        pause = input("show next? (y/n) ")
    if pause=="y":
        for pair in skeptical1:
            showInfo(pair,personDict,skeptical1[pair])
            
    pause = ""
    while pause not in ['y','n']:
        pause = input("show next? (y/n) ")
    if pause=="y":
        for pair in skeptical2:
            showInfo(pair,personDict,skeptical2[pair])