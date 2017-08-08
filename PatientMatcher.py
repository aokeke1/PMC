# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 13:35:22 2017

@author: aokeke
"""
import time, itertools,csv,operator
import numpy as np
import pickle as pkl
import Levenshtein as L
import HelperFunctions as hf

tags = ['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB',\
        'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
        'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']
"""
when looking for people of the same name and dob (findMatches2)
also considers alias (findMatches3)
"""
version = 7
def loadData(fileName,badEntries,sameThing):
    print ("Using:",fileName)
    myFile = open(fileName)
    #Prepare dictionaries for data storage
    personDict = {'EnterpriseID':{}}
    personDict2 = {'EnterpriseID':{},"LAST":{},"FIRST":{},"MIDDLE":{}}
    valueCounter = {}
    for t in tags[1:]:
        personDict[t] = {}
        valueCounter[t] = {}
    myFile.readline()
    reader = csv.reader(myFile.read().split('\n'), delimiter=',')
    for line in reader:
        #Use CSV reader to go line by line
        info = line
        if len(info)!=19:
            if len(info)!=0:
                print ("Info length:",len(info),"\ninfo:",info)
            continue
        ID = info[0]
        personDict[tags[0]][info[0]] = info
        #Make a copy with the original information because the filter may group
        #unwanted things together
        personDict2[tags[0]][info[0]] = info[:]
        
        for i in range(1,19):
            t = tags[i]
            entry = info[i]
            #badEntries like "X" or "???" or "UNK"
            if t in badEntries and entry in badEntries[t]:
                entry = ""
                info[i] = ""
            #sameThing maps a mispelled entry to its likely correct spelling
            if entry in sameThing[t]:
                entry = sameThing[t][entry]
                info[i] = entry
            
            #Deal with multiple PHONEs and ALIASes
            if t=="PHONE":
                if "^^" in entry:
                    nums = entry.split("^^")
                    entry = nums[0]
                    info[i] = entry
                    for j,n in enumerate(nums):
                        if n in personDict[t]:
                            personDict[t][n].add(ID)
                            valueCounter[t][n] += min(j,1) #the first one will be counted already because it is the main entry so set value to 0 in that case
                        else:
                            personDict[t][n] = {ID}
                            valueCounter[t][n] = min(j,1)
            if t=="PHONE2":
                t = "PHONE"
                if "^^" in entry:
                    nums = entry.split("^^")
                else:
                    nums = [entry]
                while info[15] in nums:
                    nums.remove(info[15])
                if len(nums)>0:
                    entry = nums[0]
                    info[i] = entry
                    for j,n in enumerate(nums):
                        if n in personDict[t]:
                            personDict[t][n].add(ID)
                            valueCounter[t][n] += min(j,1)
                        else:
                            personDict[t][n] = {ID}
                            valueCounter[t][n] = min(j,1)
                else:
                    entry = ""
                    info[i] = ""
            if t=="ALIAS":
                if "^^" in entry:
                    alternates = entry.split("^^")
                    entry = alternates[0]
                    info[i] = entry
                    for j,n in enumerate(alternates):
                        if n in personDict[t]:
                            personDict[t][n].add(ID)
                            valueCounter[t][n] += min(j,1)
                        else:
                            personDict[t][n] = {ID}
                            valueCounter[t][n] = min(j,1)
            
            #Save the information if it is unique
            if entry in personDict[t]:
                personDict[t][entry].add(ID)
                valueCounter[t][entry] += 1
            else:
                personDict[t][entry] = {ID}
                valueCounter[t][entry] = 1
            
            if t in ["LAST","FIRST","MIDDLE"]:
                if personDict2['EnterpriseID'][ID][i] in personDict2[t]:
                    personDict2[t][entry].add(ID)
                else:
                    personDict2[t][entry] = {ID}
            
            
    myFile.close()
    return personDict,valueCounter,personDict2



    
def findMatches(personDict,personDict2):
    """
    Find matches and skeptical matches based on EnterpriseID, SSN and PHONE
    """
    matches = {}
    skepticalMatches = {}
    for i in range(1,19):
        if tags[i] not in ['SSN','PHONE']:
            continue

        dictConsidered = personDict[tags[i]]
        done = False

        for duplicatedEntry in dictConsidered:
            if duplicatedEntry=="":
                #skip the empty entries
                continue
            pairs = itertools.combinations(dictConsidered[duplicatedEntry],2)
            if done:
                break
            for p in pairs:
                if done:
                    break

                info1 = personDict['EnterpriseID'][p[0]]
                info2 = personDict['EnterpriseID'][p[1]]
                info1b = personDict2['EnterpriseID'][p[0]]
                info2b = personDict2['EnterpriseID'][p[1]]
                k = tuple(sorted(p))
                
                if k not in matches and k not in skepticalMatches:
                    if (((info1[1]==info2[1])and info1[1]!='') or((info1[2]==info2[2])and info1[2]!='') or ((info1[5]==info2[5])and info1[5]!='') ):
                        score = getScorePair(info1b,info2b)
                        
                        
                        if (abs(int(k[0])-int(k[1]))<10) and score<7:
                            #This is likely not a real match
                            skepticalMatches[k] = score
                        else:
                            #This is a real match
                            matches[k] = score
                            
    return matches,skepticalMatches
    
def findMatches2(personDict,matches,skepticalMatches,personDict2,s2=0):
    """
    Find additonal matches using LAST, FIRST, DOB
    """
    try:
        additionalMatches = {}
        skipCount = 0
        L1 = list(personDict['LAST'])
        L2 = list(personDict['FIRST'])
        L3 = list(personDict['DOB'])
        count = 0
        for ln in L1[:]:
            count += 1
            if count%600==0:
                print (round(100*count/len(L1),3),"% complete ["+str(count)+"/"+str(len(L1))+"] after",round(time.time()-s2,2),"seconds")
                print (len(additionalMatches),"additional matches found so far...",flush=True)
            if ln=='':
                continue
            LNIDs = personDict['LAST'][ln]
            for fn in L2:
                if fn=='':
                    continue
                    
                FNIDs = personDict['FIRST'][fn]
                toPassOn = LNIDs.intersection(FNIDs)
                if len(toPassOn)==0:
                    skipCount += 1
                    continue
                    
                for dob in L3:
                    if dob=='':
                        continue
                    DOBIDs = personDict['DOB'][dob]
                    finalSet = toPassOn.intersection(DOBIDs)
                    if len(finalSet)==0:
                        skipCount += 1
                        continue
                    pairs = itertools.combinations(finalSet,2)
                    for p in pairs:
                        k = tuple(sorted(p))
                        
                        info1b = personDict2['EnterpriseID'][p[0]]
                        info2b = personDict2['EnterpriseID'][p[1]]
                        
                        if (k not in matches) and (k not in skepticalMatches) and (k not in additionalMatches):
                            badness = (L.distance(info1b[1],info2b[1])+L.distance(info1b[2],info2b[2])+2*L.distance(info1b[5],info2b[5]))
                            score = getScorePair(info1b,info2b)
                            if info1b[7]!="" and info2b[7]!="":
                                badness+=L.distance(info1b[7],info2b[7])
                            if len(info1b[12])>4 and len(info2b[12])>4:
                                if info1b[12][0:4]==info2b[12][0:4]:
                                    badness-=2
                            if badness>2 and score<5:
                                continue
                            
                            additionalMatches[k] = score
    except KeyboardInterrupt:
        return additionalMatches
    return additionalMatches

def findMatches3(personDict,matches,skepticalMatches,additionalMatches,personDict2):
    """
    Find extra matches using ALIAS
    """
    dictConsidered = personDict['ALIAS']
    for alias in dictConsidered:
        if alias == "":
            continue
        pairs = itertools.combinations(dictConsidered[alias],2)
        for p in pairs:
            k = tuple(sorted(p))
            if (k not in matches) and (k not in skepticalMatches) and (k not in additionalMatches):
                info1 = personDict['EnterpriseID'][p[0]]
                info2 = personDict['EnterpriseID'][p[1]]
                
                info1b = personDict2['EnterpriseID'][p[0]]
                info2b = personDict2['EnterpriseID'][p[1]]
                score = getScorePair(info1b,info2b)
                if score>=7:
                    additionalMatches[k] = score

    return additionalMatches

def getScorePair(info1,info2):
    """
    Given two lists of people information, calculate a score based on number of 
    matching fields.
    """
    info1 = np.array(info1)
    info2 = np.array(info2)
    score = np.count_nonzero((info1==info2) & (info1!=""))
    if info1[3]!="" and info2[3]!="" and info1[3]!=info2[3]:
        #Middle Initial vs middle name
        #Note this will count two different last names with the same first initial
        if info1[3][0]==info2[3][0]:
            score += 1
    if info1[15]!="" and info1[16]!="" and info1[15]!=info2[15] and info1[16]!=info2[16]:
        if L.distance(info1[15],info2[15])<=2 or L.distance(info1[16],info2[15])<=2 or L.distance(info1[15],info2[16])<=2 or L.distance(info1[16],info2[16])<=2:
            #if they swap primary and secondary phone numbers
            score += 1
    
    #Typos in LAST,FIRST,ALIAS. allow up to 2 mistakes
    for j in [1,2,18]:
        if info1[j]!="" and info2[j]!="" and info1[j]!=info2[j]:
            if L.distance(info1[j],info2[j])<=2:
                score += 1
            
    #Typos in DOB. allow up to 1 mistake
    if info1[5]!="" and info2[5]!="" and info1[5]!=info2[5]:
        if L.distance(info1[5],info2[5])<1:
            score += 1
    return score

def showInfo(p,personDict):
    """
    Given a tuple pair of EnterpriseIDs, displays the info for both IDs
    """
    info1 = personDict['EnterpriseID'][p[0]]
    info2 = personDict['EnterpriseID'][p[1]]
    print ("Person A:",info1)
    print ("Person B:",info2)

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
            spamwriter.writerow(list(personDict['EnterpriseID'][p[0]]))
            spamwriter.writerow(list(personDict['EnterpriseID'][p[1]]))
            spamwriter.writerow([])
            
def main(fileLabelNum=None):

#    =============================================================================
#    Section 1: Loading the data from the file and organizing it into some structures
#    =============================================================================
    start = time.time()
    print ("Version Number:",version)
    #Define file locations and which set to use
    if fileLabelNum is None:
        fileLabelNum = 1
    #Full data set
    fileName1 = "FinalDataset-1.csv"
    #100,000/1,000,000 entries
    fileName2 = "FinalDataset-1small.csv"
    #10,000/1,000,000 entries
    fileName3 = "FinalDataset-1smaller.csv"
    fileList = [fileName1,fileName2,fileName3]
    
    """
    Load in the filters
    badEntries : dictionary mapping a tag to a set of invalid options
    sameThing  : dictionary that maps a tag to a dictionary that maps an entry
                 to a different entry (used to correct typos)
    """
    badEntries,sameThing = pkl.load(open('filters.p','rb'))

    """
    Load the data and return two dictionaries. Each one has an entry for each tag (but the Phone2 tag is useless).
    personDict   : maps each tag to a dictionary that maps each unique entry to a list of ID's with the same value
    valueCounter : maps each tag to a dictionary that maps each unique entry to a count for that entry
    Takes ~50 seconds
    """
    personDict,valueCounter,personDict2 = loadData(fileList[fileLabelNum-1],badEntries,sameThing)
    print (time.time()-start,'seconds. Done loading data',flush=True)
    print ("------------------")
    print ()
    
    """
    Shows the top n values for each of the fields. Set shouldReverse to True to
    get least popular hits
    """
    hf.showTopHits(valueCounter,n=10,shouldReverse=False)
    
#    =============================================================================
#    Section 2: Finding Matches and saving found matches
#    =============================================================================


    #Output file names
    outputs1 = ['IDMatchTablev'+str(version)+'_1.csv','IDMatchTablev'+str(version)+'_2.csv','IDMatchTablev'+str(version)+'_3.csv']
    outputs2 = ['FullInfoMatchTablev'+str(version)+'_1.csv','FullInfoMatchTablev'+str(version)+'_2.csv','FullInfoMatchTablev'+str(version)+'_3.csv']
    
    """
    Find matches and skeptical matches. A dictionary that maps a pair of IDs
    to a score. Save the matches.
    Takes ~1000 seconds
    """
    matches,skepticalMatches = findMatches(personDict,personDict2)
    print (len(matches),"potential matches found.")
    print (len(skepticalMatches),"very weak matches found.")
    saveMatches(matches,personDict2,outputs1[0],outputs2[0])
    saveMatches(skepticalMatches,personDict2,outputs1[1],outputs2[1])
    
    print (time.time()-start,'seconds. Done with first set of matches.',flush=True)
    print ("------------------")
    print ()
    additionalMatches = findMatches2(personDict,matches,skepticalMatches,personDict2,s2=time.time())
    additionalMatches = findMatches3(personDict,matches,skepticalMatches,additionalMatches,personDict2)
    
    print (len(additionalMatches),"additional matches found.")
    matches.update(additionalMatches)
    print ("Total number of matches:",len(matches))
    print ("Total number of skeptical matches:",len(skepticalMatches))
    saveMatches(matches,personDict2,outputs1[1],outputs2[1])

    print (time.time()-start,'seconds. Done saving.')
    print ("------------------")
    print ()
    

    return badEntries,sameThing,personDict,valueCounter,personDict2,matches,skepticalMatches,additionalMatches

if __name__=="__main__":
    badEntries,sameThing,personDict,valueCounter,personDict2,matches,skepticalMatches,additionalMatches = main()
    pass