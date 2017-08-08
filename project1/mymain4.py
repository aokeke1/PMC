# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 23:10:37 2017

@author: aokeke
"""
import pickle as pkl
import numpy as np
import csv

pkl.dump(potentialPairs1,open("complete_pairs_list.p","wb"))

sorted_x = sorted(potentialPairs1.items(), key=operator.itemgetter(1),reverse=True)
sorted_matches = []
for i in sorted_x:
    sorted_matches.append(i[0])

with open('IDMatchTable.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['EnterpriseID1','EnterpriseID2','MATCH_SCORE'])
    for p in sorted_matches:
        spamwriter.writerow([p[0],p[1],str(potentialPairs1[p])])
    
with open('FullInfoMatchTable.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['EnterpriseID','LAST','FIRST','MIDDLE','SUFFIX','DOB','GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME','MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS'])
    for p in sorted_matches:
        spamwriter.writerow(list(listOfDicts['EnterpriseID'][p[0]]))
        spamwriter.writerow(list(listOfDicts['EnterpriseID'][p[1]]))
        spamwriter.writerow([])