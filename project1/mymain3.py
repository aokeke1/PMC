# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:44:10 2017

@author: aokeke
"""
import numpy as np
import pickle as pkl
import mymain1
import itertools
import time

np.set_printoptions(threshold=np.nan)
tags = ['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB',\
        'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
        'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']

def loadClassifiers(fileNumLabel,ext):    
    thetas = pkl.load(open(ext +"thetas"+str(fileNumLabel)+".p",'rb'))
    return thetas


def findMatches(listOfDicts,theta,theta_0):
    #Check for entries that have more than thresh things in common
    print (theta,theta_0)
    potentialPairs = {}
    for i in range(1,19):
#        if tags[i] not in ['LAST','FIRST','SSN']:
        if theta[i-1]<10:
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
                    score = np.dot(theta,(info1[1:]==info2[1:])&(info1[1:]!=""))+theta_0
                    if score>0:
#                        print (score)
                        potentialPairs[k] = score
                        
#                    else:
#                        print ("removed")
#        print (len(potentialPairs),"potential pairs so far",flush=True)
            
    return potentialPairs

def showMatches(potentialPairs,listOfDicts):
    for p in potentialPairs:
        print ("The following pair has a score of",potentialPairs[p],":")
        print ("Person A:",listOfDicts['EnterpriseID'][p[0]])
        print ("Person B:",listOfDicts['EnterpriseID'][p[1]])
        print ("--------------------")

if __name__=="__main__":
    startTime = time.time()
    #Load classifiers
    fileNumLabel = 2
#    ext = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/"
#    if fileNumLabel ==2:
#        ext+="small/"
#    elif fileNumLabel ==3:
#        ext+="smaller/"
#        
#    thetas = loadClassifiers(fileNumLabel,ext)
    thetas = pkl.load(open("thetas.p",'rb'))
    (best_theta_1,best_theta_0_1),(best_theta_2,best_theta_0_2),(best_theta_3,best_theta_0_3) = thetas
    
    #Load and process full data
    fileName1 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/FinalDataset-1.csv"
    fileName2 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/small/FinalDataset-1small.csv"
    fileName3 = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/smaller/FinalDataset-1smaller.csv"
    
    listOfDicts = mymain1.getInfoIntoDicts(fileName1)
    listOfDicts = mymain1.dataProcessing(listOfDicts)
    
    #Classify data
    potentialPairs1 = findMatches(listOfDicts,best_theta_2,best_theta_0_2)
    print (len(potentialPairs1),"matches found")
    
#    potentialPairs2 = findMatches(listOfDicts,best_theta_2,best_theta_0_2)
#    print (len(potentialPairs2),"matches found")
#    
#    potentialPairs3 = findMatches(listOfDicts,best_theta_3,best_theta_0_3)
#    print (len(potentialPairs3),"matches found")
    #Display results
#    showMatches(potentialPairs,listOfDicts)
    pass

    endTime = time.time()
    print ("This process took ",(endTime-startTime)," seconds")