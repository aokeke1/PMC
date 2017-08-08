# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:40:33 2017

@author: aokeke
"""
import numpy as np
import operator
import Levenshtein as L
import copy
np.set_printoptions(threshold=np.nan)
tags = ['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB',\
        'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
        'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']

def showTopHits(valueCounter,n=10,shouldReverse=False):
    for i in range(1,19):
        t = tags[i]
        print ("Tag:",t)

        dictConsidered = valueCounter[t].copy()
        if "" in dictConsidered:
            dictConsidered.pop('')
        
        sorted_x = sorted(dictConsidered.items(), key=operator.itemgetter(1),reverse=not shouldReverse)
        print (sorted_x[:n])
        print ("------------------")
        
        
def findBads(valueCounter,badEntries,shouldReverse=False):
#    badEntries = {}
    n = 10
    for i in range(1,19):
        t = tags[i]
        print ("Tag:",t)
        dictConsidered = valueCounter[t].copy()
        dictConsidered.pop('')
        
        sorted_x = sorted(dictConsidered.items(), key=operator.itemgetter(1),reverse=not shouldReverse)

#        print (sorted_x[:n])
        if t not in badEntries:
            badEntries[t] = set()
        for j,key_value in enumerate(sorted_x):
            print ("Queue:",sorted_x[j:j+n])
            print ("Current:",key_value)
            shouldDiscard = ""
            while shouldDiscard not in ['y','n','0']:
                shouldDiscard = input("Is this bad? (y/n) 0 to continue to next entry ")
                
            if shouldDiscard=="0":
                break
            elif shouldDiscard=="y":
                badEntries[t].add(key_value[0])
            elif shouldDiscard=="n":
                pass
            
            
        print ("------------------")
    return badEntries

def matchSameGroup(personDict,valueCounter,t,numOfVals,maxDist,sameThing):
    """
    check if the first numOfVals entries in the valueCounter for a specific tag
    has any similar entries. Similar meaning within maxDist Levenshtein distance.
    """
    print ("Tag:",t)
    dictConsidered = valueCounter[t].copy()
    dictConsidered.pop('')
    
    sorted_x = sorted(dictConsidered.items(), key=operator.itemgetter(1),reverse=True)
    allEntries = []
    for p in sorted_x:
        allEntries.append(p[0])
    print (len(allEntries),"different values for this entry.")
    for e in allEntries[:numOfVals]:
        if e in sameThing[t]:
            continue
        pause = input("Pause")
        if pause=="0":
            return sameThing
        print ("Entries similar to",e,":")
        for e2 in allEntries:
            if e2==e or e2 in sameThing[t]:
                continue
            if L.distance(e,e2)<=maxDist:
                print (e2)
                isSame = ""
                while isSame not in ['y','n','0']:
                    isSame = input("Is this the same as the super? (y/n)?")
                if isSame=='y':
                    sameThing[t][e2] = e
                elif isSame=='0':
                    break
        print ("------------------")
    return sameThing
    
def matchSameGroupAuto(personDict,valueCounter,sameThing):
    """
    Use difference dict to group together similar entries for a filter.p file
    """
    sameThing2 = copy.deepcopy(sameThing)
    masterValues = {}
    diffDict = {'LAST':3,'FIRST':3,'MIDDLE':3,'DOB':1,'ADDRESS1':6,'ZIP':1,'MOTHERS_MAIDEN_NAME':2,'CITY':2,'ALIAS':3}
    for i in range(1,19):
        t = tags[i]
        if t=="GENDER" or t=="PHONE2" or t=="SUFFIX" or t=="SSN" or t=="PHONE" or t=="ADDRESS2" or t=="EMAIL" or t=="STATE" or t=="MRN" or t=="ADDRESS1":
            continue
        print ("Tag:",t)
        dictConsidered = copy.deepcopy(valueCounter[t])
        dictConsidered.pop('')
        
        sorted_x = sorted(dictConsidered.items(), key=operator.itemgetter(1),reverse=True)
        allEntries = []
        for p in sorted_x:
            allEntries.append(p[0])
        
        masterValues[t] = set(sameThing2[t].values())
            
        print (len(allEntries),"different values for this entry.")
        count = 0
        for e in allEntries:
            count += 1
            if count%int(len(allEntries)/10)==0:
                print (round(100*count/len(allEntries),2),"% complete")
            if (e in sameThing2[t]) or (e in masterValues[t]):
                #already has a group or is a group leader
                continue
            noGroup = True
            if len(e)>diffDict[t]+1:
                for e2 in masterValues[t]:
                    if len(e2)<=diffDict[t]+2:
                        continue
                        
                    if L.distance(e,e2)<=diffDict[t]:
                        sameThing2[t][e] = e2
                        noGroup = False
                        break
            if noGroup:
                masterValues[t].add(e)
        print (len(masterValues[t]),"different values for this entry after pruning.")
        print ('----------------------')
    return sameThing2,masterValues