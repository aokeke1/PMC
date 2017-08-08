# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 09:29:42 2017

@author: aokeke
"""

import project1 as p1
import utils
import numpy as np
import pickle as pkl

##-------------------------------------------------------------------------------
##  Section 1
## Data loading.
##-------------------------------------------------------------------------------
#
#fileNumLabel = 2
#ext = "C:/Users/aokeke/Documents/InterSystems/CacheEnsemble/PatientMatchingChallenge/"
#if fileNumLabel ==2:
#    ext+="small/"
#elif fileNumLabel ==3:
#    ext+="smaller/"
#train_bow_features = pkl.load(open(ext +"trainData"+str(fileNumLabel)+".p",'rb'))
#val_bow_features = pkl.load(open(ext +"valData"+str(fileNumLabel)+".p",'rb'))
#test_bow_features = pkl.load(open(ext +"testData"+str(fileNumLabel)+".p",'rb'))
#
#train_labels = pkl.load(open(ext +"trainLabels"+str(fileNumLabel)+".p",'rb'))
#val_labels = pkl.load(open(ext +"valLabels"+str(fileNumLabel)+".p",'rb'))
#test_labels = pkl.load(open(ext +"testLabels"+str(fileNumLabel)+".p",'rb'))
#
#reference = pkl.load(open(ext +"reference"+str(fileNumLabel)+".p",'rb'))
#    
#print ("(train_bow_features.shape,val_bow_features.shape,test_bow_features.shape)",(train_bow_features.shape,val_bow_features.shape,test_bow_features.shape))
fileName = "C:/Users/aokeke/Documents/GITHUB/PMC/MLdata.pkl"

((train_bow_features,train_labels),(val_bow_features,val_labels),(test_bow_features,test_labels),reference) =pkl.load(open(fileName,'rb'))
##-------------------------------------------------------------------------------
## Section 2
## A first glance at the data
##-------------------------------------------------------------------------------
def tryDataOut(train_bow_features,val_bow_features,test_bow_features,train_labels,val_labels,test_labels,reference,T=(10,10,10),L=0.01):
    pct_train_accuracy, pct_val_accuracy = \
        p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T[0],reference=reference)
    print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
    print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
    print ("------------------------------------")
    avg_pct_train_accuracy, avg_pct_val_accuracy = \
        p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T[1],reference=reference)
    print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
    print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))
    print ("------------------------------------")
    avg_peg_train_accuracy, avg_peg_val_accuracy = \
        p1.pegasos_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T[2],L=L,reference=reference)
    print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
    print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
##-------------------------------------------------------------------------------
##
##
##-------------------------------------------------------------------------------
## Section 3
## Use validation set to fit the hyper parameters
##-------------------------------------------------------------------------------

def trainWithValidation(train_bow_features, train_labels, val_bow_features, val_labels):
    data = (train_bow_features, train_labels, val_bow_features, val_labels)
    
    # values of T and lambda to try
    #Ts = [1, 5, 10, 15, 25, 50, 100]
    #Ls = [0.01, 0.1, 0.2, 0.5, 1]

    Ts = range(1,100)
    Ls = np.linspace(0.001,1,100)
    
    pct_tune_results = utils.tune_perceptron(Ts, *data)
    best_T_pct = Ts[list(pct_tune_results[1]).index(max(pct_tune_results[1]))]
    print ("Perceptron best T =",best_T_pct)
    
    avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
    best_T_avgpct = Ts[list(avg_pct_tune_results[1]).index(max(avg_pct_tune_results[1]))]
    print ("Average Perceptron best T =",best_T_avgpct)
    
    # fix values for L and T while tuning Pegasos T and L, respective
    best_L = 0.001
    best_T = 100
    
    avg_peg_tune_results_T = utils.tune_pegasos_T(best_L, Ts, *data)
    best_T = Ts[list(avg_peg_tune_results_T[1]).index(max(avg_peg_tune_results_T[1]))]
    print ("Pegasos best T =",best_T)
    avg_peg_tune_results_L = utils.tune_pegasos_L(best_T, Ls, *data)
    best_L = Ls[list(avg_peg_tune_results_L[1]).index(max(avg_peg_tune_results_L[1]))]
    print ("Pegasos best L =",best_L)
    
    utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
    utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
    utils.plot_tune_results('Pegasos', 'T', Ts, *avg_peg_tune_results_T)
    utils.plot_tune_results('Pegasos', 'L', Ls, *avg_peg_tune_results_L)
    
    return best_T_pct,best_T_avgpct,(best_T,best_L)
##-------------------------------------------------------------------------------
##
##-------------------------------------------------------------------------------
## Section 4
##
## Get the best classifiers from the found best hyperparameters
##-------------------------------------------------------------------------------
def trainAndTest(train_bow_features,val_bow_features,test_bow_features,train_labels,val_labels,test_labels,reference,T=(10,10,10),L=0.01,algo=0):
    if algo in range(3):
        if algo==0:
            #Perceptron
            pct_train_accuracy, pct_val_accuracy = \
                p1.perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[0],reference=reference,s2='test')
            print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
            print("{:35} {:.4f}".format("Test accuracy for perceptron:", pct_val_accuracy))
            best_theta,best_theta_0 = p1.perceptron(train_bow_features, train_labels, T[0])
        elif algo==1:
            #Average Perceptron
            avg_pct_train_accuracy, avg_pct_val_accuracy = \
                p1.average_perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[1],reference=reference,s2='test')
            print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
            print("{:43} {:.4f}".format("Test accuracy for average perceptron:", avg_pct_val_accuracy))
            best_theta,best_theta_0 = p1.average_perceptron(train_bow_features, train_labels, T[1])
        elif algo==2:
            #Pegasos
            avg_peg_train_accuracy, avg_peg_val_accuracy = \
                p1.pegasos_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[2],L=L,reference=reference,s2='test')
            print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
            print("{:50} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_val_accuracy))
            best_theta,best_theta_0 = p1.pegasos(train_bow_features, train_labels, T[2], L)
        return best_theta,best_theta_0
    elif algo==3:
        #All 3
        pct_train_accuracy, pct_val_accuracy = \
            p1.perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[0],reference=reference,s2='test')
        print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
        print("{:35} {:.4f}".format("Test accuracy for perceptron:", pct_val_accuracy))
        print ("------------------------------------")
        avg_pct_train_accuracy, avg_pct_val_accuracy = \
            p1.average_perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[1],reference=reference,s2='test')
        print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
        print("{:43} {:.4f}".format("Test accuracy for average perceptron:", avg_pct_val_accuracy))
        print ("------------------------------------")
        avg_peg_train_accuracy, avg_peg_val_accuracy = \
            p1.pegasos_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=T[2],L=L,reference=reference,s2='test')
        print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
        print("{:50} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_val_accuracy))
        
        best_theta_1,best_theta_0_1 = p1.perceptron(train_bow_features, train_labels, T[0])
        best_theta_2,best_theta_0_2 = p1.average_perceptron(train_bow_features, train_labels, T[1])
        best_theta_3,best_theta_0_3 = p1.pegasos(train_bow_features, train_labels, T[2], L)


        return (best_theta_1,best_theta_0_1),(best_theta_2,best_theta_0_2),(best_theta_3,best_theta_0_3)

##-------------------------------------------------------------------------------
##
##-------------------------------------------------------------------------------
## Section 5
##
## Learn from an algorithm which properties were most helpful in classifying points
##-------------------------------------------------------------------------------


def findMostDecisiveTags(best_theta,numToShow=5):
    tags = ['LAST','FIRST','MIDDLE','SUFFIX','DOB',\
            'GENDER','SSN','ADDRESS1','ADDRESS2','ZIP','MOTHERS_MAIDEN_NAME',\
            'MRN','CITY','STATE','PHONE','PHONE2','EMAIL','ALIAS']

    sorted_word_features = utils.most_explanatory_word(best_theta, tags)
    print("Most Explanatory Word Features")
    #sorted_word_features.reverse()
    print(sorted_word_features[:numToShow])
    return sorted_word_features[:numToShow]

##-------------------------------------------------------------------------------
##
##-------------------------------------------------------------------------------
## Section 6
##
## Call all the previously define algorithms
##-------------------------------------------------------------------------------

if __name__=="__main__":
    #Get a sense of how good it will work
    print ("A first glance at the data...")
    tryDataOut(train_bow_features,val_bow_features,test_bow_features,train_labels,val_labels,test_labels,reference)
    
    print ()
    print ("Training the hyper parameters...")
    best_T_pct,best_T_avgpct,(best_T,best_L) = trainWithValidation(train_bow_features, train_labels, val_bow_features, val_labels)
    
    print ()
    print ("Final training and testing...")
#    thetas = trainAndTest(train_bow_features,val_bow_features,test_bow_features,train_labels,val_labels,test_labels,reference,T=(best_T_pct,best_T_avgpct,best_T),L=best_L,algo=3)
    thetas = trainAndTest(train_bow_features,val_bow_features,test_bow_features,train_labels,val_labels,test_labels,reference,T=(30,30,30),L=0.01,algo=3)
    (best_theta_1,best_theta_0_1),(best_theta_2,best_theta_0_2),(best_theta_3,best_theta_0_3) = thetas
    
    print ()
    print ("Useful tag identifications...")
    print ("Perceptron:")
    bestWords_1 = findMostDecisiveTags(best_theta_1,numToShow=5)
    print ("Averaged Perceptron:")
    bestWords_1 = findMostDecisiveTags(best_theta_2,numToShow=5)
    print ("Pegasos:")
    bestWords_1 = findMostDecisiveTags(best_theta_3,numToShow=5)
    
    
    
    print ()
    print ()
    print ("-----------------------------------")
    print ("Perceptron Parameters:\ntheta =",best_theta_1,"\ntheta_0 =",best_theta_0_1)
    print ("-----------------------------------")
    print ("Averaged Perceptron Parameters:\ntheta =",best_theta_2,"\ntheta_0 =",best_theta_0_2)
    print ("-----------------------------------")
    print ("Pegasos Parameters:\ntheta =",best_theta_3,"\ntheta_0 =",best_theta_0_3)
    
    pkl.dump(thetas,open("thetas.p","wb"))