from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)
#np.random.seed(0)

### Part I

def hinge_loss(feature_matrix, labels, theta, theta_0):
    """
    Section 1.2
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    h = 0
    for i in range(len(feature_matrix)):
        h += max(0,1-labels[i]*(theta.dot(feature_matrix[i])+theta_0))
    return h/len(feature_matrix)

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Section 1.3
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
#    print (current_theta)
    if label*(feature_vector.dot(current_theta) + current_theta_0)<=0:
        current_theta += label*feature_vector
        current_theta_0 += label
    return (current_theta,current_theta_0)

def perceptron(feature_matrix, labels, T):
    """
    Section 1.4a
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    temp = np.hstack((feature_matrix,np.array(labels).reshape(len(labels),1)))
    np.random.shuffle(temp)
    feature_matrix = temp[:,0:-1]
    labels = temp[:,-1]
    current_theta = np.zeros(len(feature_matrix[0]))
    current_theta_0 = 0
    for t in range(T):
        for i in range(len(labels)):
            feature_vector = feature_matrix[i]
            label = labels[i]
            current_theta,current_theta_0 = perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0)
    return (current_theta,current_theta_0)
#    raise NotImplementedError
    
def average_perceptron(feature_matrix, labels, T):
    """
    Section 1.4b
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    temp = np.hstack((feature_matrix,np.array(labels).reshape(len(labels),1)))
    np.random.shuffle(temp)
    feature_matrix = temp[:,0:-1]
    labels = temp[:,-1]

    current_theta = np.zeros(len(feature_matrix[0]))
    current_theta_0 = 0
    avg_theta = np.zeros(len(feature_matrix[0]))
    avg_theta_0 = 0
    for t in range(T):
        for i in range(len(labels)):
            feature_vector = feature_matrix[i]
            label = labels[i]
            current_theta,current_theta_0 = perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0)
            avg_theta += current_theta
            avg_theta_0 += current_theta_0
    avg_theta = avg_theta/(len(labels)*T)
    avg_theta_0 = avg_theta_0/(len(labels)*T)
    return (avg_theta,avg_theta_0)
    
#    raise NotImplementedError

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label*(current_theta.dot(feature_vector)+current_theta_0)<=1:
        current_theta = (1-L*eta)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
    else:
        current_theta = (1-L*eta)*current_theta

    return (current_theta,current_theta_0)
#    raise NotImplementedError

def pegasos(feature_matrix, labels, T, L):
    """
    Section 1.6
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    
    For each update, set learning rate = 1/sqrt(t), 
    where t is a counter for the number of updates performed so far (between 1 
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    temp = np.hstack((feature_matrix,np.array(labels).reshape(len(labels),1)))
    np.random.shuffle(temp)
    feature_matrix = temp[:,0:-1]
    labels = temp[:,-1]
    
    t = 0
    current_theta = np.zeros(len(feature_matrix[0]))
    current_theta_0 = 0
    for j in range(T):
        for i in range(len(labels)):
            t+=1
            eta = 1/(np.sqrt(t))
            label = labels[i]
            feature_vector = feature_matrix[i]
            current_theta, current_theta_0 = pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0)
    return (current_theta,current_theta_0)
#    raise NotImplementedError

### Part II

def classify(feature_matrix, theta, theta_0):
    """
    Section 2.8
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    labels = np.zeros(len(feature_matrix))
    for i in range(len(feature_matrix)):
        value = np.sign(theta.dot(feature_matrix[i])+theta_0)
        if abs(value)<=1e-4:
            value = 1
        labels[i] = value
    return labels
#    raise NotImplementedError

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T,reference,s1="train",s2="val"):
    """
    Section 2.9
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    
    theta,theta_0 = perceptron(train_feature_matrix, train_labels, T)
    print("theta:",theta,"\ntheta_0:",theta_0)
    trainClass = classify(train_feature_matrix, theta, theta_0)
    valClass = classify(val_feature_matrix, theta, theta_0)
    
    trainAcc = accuracy(trainClass, np.asarray(train_labels))
    valAcc = accuracy(valClass, val_labels)
    
    show_errors(trainClass,train_labels,reference,"Perceptron",s1)
    show_errors(valClass,val_labels,reference,"Perceptron",s2)
    
    return (trainAcc,valAcc)
    
#    raise NotImplementedError

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T,reference,s1="train",s2="val"):
    """
    Section 2.9
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    
    theta,theta_0 = average_perceptron(train_feature_matrix, train_labels, T)
    print("theta:",theta,"\ntheta_0:",theta_0)
    trainClass = classify(train_feature_matrix, theta, theta_0)
    valClass = classify(val_feature_matrix, theta, theta_0)
    
    trainAcc = accuracy(trainClass, train_labels)
    valAcc = accuracy(valClass, val_labels)
    
    show_errors(trainClass,train_labels,reference,"Averaged Perceptron",s1)
    show_errors(valClass,val_labels,reference,"Averaged Perceptron",s2)
    
    return (trainAcc,valAcc)
    
#    raise NotImplementedError

def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L,reference,s1="train",s2="val"):
    """
    Section 2.9
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    
    theta,theta_0 = pegasos(train_feature_matrix, train_labels, T, L)
    print("theta:",theta,"\ntheta_0:",theta_0)
    trainClass = classify(train_feature_matrix, theta, theta_0)
    valClass = classify(val_feature_matrix, theta, theta_0)
    
    trainAcc = accuracy(trainClass, train_labels)
    valAcc = accuracy(valClass, val_labels)
    
    show_errors(trainClass,train_labels,reference,"Pegasos",s1)
    show_errors(valClass,val_labels,reference,"Pegasos",s2)
    return (trainAcc,valAcc)

def show_errors(classifications,labels,reference,algoName,source):
#    return
    print ("Showing misclassified pairs for algorithm:",algoName)
    print ("Source:",source)
    for i in range(len(classifications)):
        if classifications[i]!=labels[i]:
            if np.random.rand()>=0.01:
                continue
            print (reference[source][i])
            print ("Actual Label:",labels[i])
            print ("Given Label:",classifications[i])

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    punctuation + digits
    for c in punctuation + digits:
        
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_words2(input_string):
    """
    Helper function for bag_of_words2()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are removed all together.
    """
    
    for c in punctuation + digits:
        
        input_string = input_string.replace(c, ' ' + c + ' ')
        #Used for removing punctuation
#        if c not in [':',')','!']:
#            input_string = input_string.replace(c, '')
#        input_string = input_string.replace(c, '')

        
    wordList = input_string.lower().split()
    
    #Used for filtering stop words
#    stopWords = []
#    f = open('stopwords.txt', 'r')
#    for line in f:
#        stopWords.append(line[:-1])
#    f.close()
#
#    newWordList = []
#    for word in wordList:
#        if word not in stopWords:
#            newWordList.append(word)
#    return newWordList
    return wordList

def bag_of_words2(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    n = 2 #max length of phrases
    dictionary = {} # maps word to unique index
    foundWords = {} #for keeping track of how many times a word appeared. e.g. words only enter the dictionary if more than two of them appear in the text
    for text in texts:
        word_list = extract_words2(text)
#        for word in word_list:
#            if word not in dictionary:
#                dictionary[word] = len(dictionary)
#        print ("number of words:",len(word_list))
        for j in range(1,n+1):
#            print ("j:",j)
            for i in range(len(word_list)-(j-1)):
#                print ("i:",j)
                newWord = ""
                for k in range(j):
                    newWord = newWord + " " + word_list[k+i]
                newWord = newWord[1:]
                if newWord not in foundWords:
                    foundWords[newWord] = 1
                else:
                    foundWords[newWord] = foundWords[newWord]+1
                if newWord not in dictionary and foundWords[newWord]>=2:
                    dictionary[newWord] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

def extract_bow_feature_vectors2(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words2(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1


    return feature_matrix

def extract_additional_features(reviews, dictionary):
    """
    Section 3.12
    Inputs a list of string reviews
    Returns a feature matrix of (n,m), where n is the number of reviews
    and m is the total number of additional features of your choice

    YOU MAY CHANGE THE PARAMETERS
    """

    num_reviews = len(reviews)
    print ("num of words:",len(dictionary),flush=True)
    print ("num of reviews:",num_reviews,flush=True)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])
    n=2
    for i, text in enumerate(reviews):
        word_list = extract_words2(text)
#        for word in word_list:
#            if word in dictionary:
#                feature_matrix[i, dictionary[word]] = 1
        for j in range(2,n+1): #2 because original extract does length one words already
            for m in range(len(word_list)-(j-1)):
                newWord = ""
                for k in range(j):
                    newWord = newWord + " " + word_list[k+m]
                newWord = newWord[1:]
                if newWord in dictionary:
                    feature_matrix[i, dictionary[newWord]] = 1
                        
    return feature_matrix
    
#    return np.ndarray((len(reviews), 0))

def extract_final_features(reviews, dictionary):
    """
    Section 3.12
    Constructs a final feature matrix using the improved bag-of-words and/or additional features
    """
    bow_feature_matrix = extract_bow_feature_vectors2(reviews,dictionary)
    additional_feature_matrix = extract_additional_features(reviews,dictionary)
    return np.hstack((bow_feature_matrix, additional_feature_matrix))

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
