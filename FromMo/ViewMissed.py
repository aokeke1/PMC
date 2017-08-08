# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:04:43 2017

@author: aokeke
"""
import csv

def showInfo(p,personDict):
    """
    Given a tuple pair of EnterpriseIDs, displays the info for both IDs
    """
    info1 = personDict[p[0]]
    info2 = personDict[p[1]]
    print ("Person A:",info1)
    print ("Person B:",info2)
    
def loadPeople():
    personDict = {}
    
    fileNames = ["FinalDataset-1.csv","FinalDataset-2.csv"]
#    ext = "C:/Users/aokeke/Documents/GITHUB/PMC/"
    ext = "C:/Users/arinz/Desktop/2016-2017/Projects/PatientMatchingChallenge/PMC/"
    for name in fileNames:
        #Load matches
        
        fileName = ext + name
        myFile = open(fileName)
        myFile.readline()
        reader = csv.reader(myFile.read().split('\n'), delimiter=',')
        for line in reader:
            if len(line)>0:
                personDict[line[0]] = line
        myFile.close()
    return personDict
    
def viewMissed():
    missed = set()
#    personDict = loadPeople()
    fileName = "CACHETRM.LOG"
    
    myFile = open(fileName)
    
    for line in myFile:
        p = line.split(",")
#        print ("p: ",p)
        if len(p)!=2:
            continue
        p[0] = p[0].replace("\n","")
        p[0] = p[0].replace("\x00","")
        p[0] = p[0].replace("ÿþ","")
        p[1] = p[1].replace("\n","")
        p[1] = p[1].replace("\x00","")
        p[1] = p[1].replace("ÿþ","")
#        print ("p: ",p)
        k = tuple(sorted(p))
        missed.add(k)
#        showInfo(k,personDict)
#        pause = input("continue? (y/n)")
#        if pause=="n":
#            break
#        
    myFile.close()
    return missed

def loadMatches():
    matched = {}
    
    fileName1 = "C:/Users/arinz/Desktop/2016-2017/Projects/PatientMatchingChallenge/PMC/MRNIDMatchesCutoff5.csv"

    #Load matches
    myFile = open(fileName1)
    myFile.readline()
    reader = csv.reader(myFile.read().split('\n'), delimiter=',')
    for line in reader:
        if len(line)>0:
            matched[(line[0],line[1])] = float(line[2])
        
    return matched

def compare(missed,matched):
    personDict = loadPeople()
    for p in missed:
        if p in matched :
            matched.pop(p)
    print (len(matched),"remaining pairs missed")
    pause = input("show? (y/n) ")
    if pause=="y":
        for p in matched:
            showInfo(p,personDict)
            pause = input("continue? (y/n)")
            if pause=="n":
                break
    return

if __name__ == "__main__":
    pass
    missed = viewMissed()
    matched = loadMatches()
    compare(missed,matched)
#    missed = viewMissed()