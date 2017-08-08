import project1 as p1
import utils

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Section 1.7
#-------------------------------------------------------------------------------
toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

T = 10
L = 0.2

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)

def plot_toy_results(algo_name, thetas):
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)
#-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.9.b
#-------------------------------------------------------------------------------
print ("Original Function")
T = 10
L = 0.01

pct_train_accuracy, pct_val_accuracy = \
    p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
    p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
    p1.pegasos_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
##-------------------------------------------------------------------------------
#
#
#-------------------------------------------------------------------------------
# Section 2.10
#-------------------------------------------------------------------------------
data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
#Ts = [1, 5, 10, 15, 25, 50, 100]
#Ls = [0.01, 0.1, 0.2, 0.5, 1]

import numpy as np
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
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11a
#
# Call one of the accuracy functions that you wrote in part 2.9.a and report
# the hyperparameter and accuracy of your best classifier on the test data.
# The test data has been provided as test_bow_features and test_labels.
#-------------------------------------------------------------------------------
best_T_pct = 45
best_T_avgpct = 15
best_T = 89
best_L = 0.001

pct_train_accuracy, pct_val_accuracy = \
    p1.perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=best_T_pct)
pct_train_accuracy, pct_test_accuracy = \
    p1.perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=best_T_pct)
    
print ("Best T =",best_T_pct)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))
print("{:35} {:.4f}".format("Test accuracy for perceptron:", pct_test_accuracy))



avg_pct_train_accuracy, avg_pct_val_accuracy = \
    p1.average_perceptron_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=best_T_avgpct)

avg_pct_train_accuracy, avg_pct_test_accuracy = \
    p1.average_perceptron_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=best_T_avgpct)
    
print ("Best T =",best_T_avgpct)
print("{:43} {:.4f}".format("Training accuracy for Average Perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for Average Perceptron:", avg_pct_val_accuracy))
print("{:43} {:.4f}".format("Test accuracy for Average Perceptron:", avg_pct_test_accuracy))


avg_peg_train_accuracy, avg_peg_val_accuracy = \
    p1.pegasos_accuracy(train_bow_features,val_bow_features,train_labels,val_labels,T=best_T,L=best_L)

avg_peg_train_accuracy, avg_peg_test_accuracy = \
    p1.pegasos_accuracy(train_bow_features,test_bow_features,train_labels,test_labels,T=best_T,L=best_L)
    
print ("Best T =",best_T,"Best L =",best_L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
print("{:50} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 2.11b
#
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------
best_T_avgpct = 15
best_theta,best_theta_0 = p1.average_perceptron(train_bow_features, train_labels, best_T_avgpct)
#best_theta = None
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
#sorted_word_features.reverse()
print(sorted_word_features[:10])
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.12
#
# After constructing a final feature representation, use code similar to that in
# sections 2.9b and 2.10 to assess its performance on the validation set.
# You may use your best classifier from before as a baseline.
# When you are satisfied with your features, evaluate their accuracy on the test
# set using the same procedure as in section 2.11a.
#-------------------------------------------------------------------------------
print ("Modified Function")
dictionary2 = p1.bag_of_words2(train_texts)

train_final_features = p1.extract_final_features(train_texts, dictionary2)
val_final_features   = p1.extract_final_features(val_texts, dictionary2)
test_final_features  = p1.extract_final_features(test_texts, dictionary2)

best_T_pct = 45
best_T_avgpct = 15
best_T = 89
best_L = 0.01

theta,theta_0 = p1.perceptron(train_final_features, train_labels, best_T_pct)

trainClass = p1.classify(train_final_features, theta, theta_0)
valClass = p1.classify(val_final_features, theta, theta_0)
testClass = p1.classify(test_final_features, theta, theta_0)

trainAcc = p1.accuracy(trainClass, train_labels)
valAcc = p1.accuracy(valClass, val_labels)
testAcc = p1.accuracy(testClass, test_labels)

print("{:35} {:.4f}".format("Training accuracy for perceptron:", trainAcc))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", valAcc))
print("{:35} {:.4f}".format("Test accuracy for perceptron:", testAcc))


theta,theta_0 = p1.average_perceptron(train_final_features, train_labels, best_T_avgpct)

trainClass = p1.classify(train_final_features, theta, theta_0)
valClass = p1.classify(val_final_features, theta, theta_0)
testClass = p1.classify(test_final_features, theta, theta_0)

trainAcc = p1.accuracy(trainClass, train_labels)
valAcc = p1.accuracy(valClass, val_labels)
testAcc = p1.accuracy(testClass, test_labels)

print("{:43} {:.4f}".format("Training accuracy for average perceptron:", trainAcc))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", valAcc))
print("{:43} {:.4f}".format("Test accuracy for average perceptron:", testAcc))


theta,theta_0 = p1.pegasos(train_final_features, train_labels, best_T, best_L)

trainClass = p1.classify(train_final_features, theta, theta_0)
valClass = p1.classify(val_final_features, theta, theta_0)
testClass = p1.classify(test_final_features, theta, theta_0)

trainAcc = p1.accuracy(trainClass, train_labels)
valAcc = p1.accuracy(valClass, val_labels)
testAcc = p1.accuracy(testClass, test_labels)

print("{:50} {:.4f}".format("Training accuracy for Pegasos:", trainAcc))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", valAcc))
print("{:50} {:.4f}".format("Test accuracy for Pegasos:", testAcc))
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
# Section 3.13
#
# Modify the code below to extract your best features from the submission data
# and then classify it using your most accurate classifier.
#-------------------------------------------------------------------------------
print ("Analyzing reviews_submit")
submit_texts = [sample['text'] for sample in utils.load_data('reviews_submit.tsv')]

# 1. Extract your preferred features from the train and submit data
#dictionary = p1.bag_of_words(submit_texts)
dictionary = p1.bag_of_words2(train_texts)
train_final_features = p1.extract_final_features(train_texts, dictionary)
submit_final_features = p1.extract_final_features(submit_texts, dictionary)

# 2. Train your most accurate classifier
# final_thetas = p1.perceptron(train_final_features, train_labels, T=1)
final_thetas = p1.pegasos(train_final_features, train_labels, best_T, best_L)

# 3. Classify and write out the submit predictions.
submit_predictions = p1.classify(submit_final_features, *final_thetas)
utils.write_predictions('reviews_submit.tsv', submit_predictions)
#-------------------------------------------------------------------------------

