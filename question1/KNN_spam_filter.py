import os
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# Load the data

def load_data():
    print("Loading data...")
    
    ham_files_location = os.listdir("dataset/ham")
    spam_files_location = os.listdir("dataset/spam")
    data = []
    
    # Load ham email
    for file_path in ham_files_location:
        f = open("dataset/ham/" + file_path, "r")
        text = str(f.read())
        data.append([text, "ham"])
    
    # Load spam email
    for file_path in spam_files_location:
        f = open("dataset/spam/" + file_path, "r")
        text = str(f.read())
        data.append([text, "spam"])
        
    data = np.array(data)
    
    print("flag 1: loaded data")
    return data


# Preprocessing data: noise removal

def preprocess_data(data):
    print("Preprocessing data...")
    
    punc = string.punctuation           # Punctuation list
    sw = stopwords.words('english')     # Stopwords list
    
    for record in data:
        # Remove common punctuation and symbols
        for item in punc:
            record[0] = record[0].replace(item, "")
             
        # Lowercase all letters and remove stopwords 
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word  # Takes back all non-stopwords
        record[0] = newText
        
    print("flag 2: preprocessed data")        
    return data


# Splitting original dataset into training dataset and test dataset

def split_data(data):
    print("Splitting data...")
    
    features = data[:, 0]   # array containing all email text bodies
    labels = data[:, 1]     # array containing all corresponding labels
    
    training_data, test_data, training_labels, test_labels =\
        train_test_split(features, labels, test_size = 0.20, random_state = 42)
    
    print("flag 3: splitted data")
    return training_data, test_data, training_labels, test_labels


# Get count of each word in email, returns dictionary with each word with the respective count

def get_count(text):
    wordCounts = dict()
    for word in text.split():
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    
    return wordCounts


# Get the similarity by calculating the euclidean difference, returns double

def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0
     
    for word in test_WordCounts:
        # if word is in both emails, calculate count difference, square it, and add to total
        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word])**2
            
            # to remove common words, to speed up processing in next for loop
            del training_WordCounts[word] 
            
        # if word in test email only, square the count and add to total
        else:
            total += test_WordCounts[word]**2
    
    # Square the count of words only in training email and add to total
    for word in training_WordCounts:
            total += training_WordCounts[word]**2
            
    return total**0.5


# Returns the class to which current email belongs to as a string

def get_class(selected_Kvalues):
    spam_count = 0
    ham_count = 0
    
    # Counts the frequency of each class in K nearest neighbours
    for value in selected_Kvalues:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1
    
    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"
    
                       
# KNN classifier, returns list of labels of test_data

def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Running KNN Classifier...")
    
    result = []
    counter = 1
    
    # word counts for training email
    training_WordCounts = [] 
    for training_text in training_data:
            training_WordCounts.append(get_count(training_text))  
            
    for test_text in test_data:
        similarity = [] # List of euclidean distances
        test_WordCounts = get_count(test_text)  # word counts for test email
        
        # Getting euclidean difference 
        for index in range(len(training_data)):
            euclidean_diff =\
                euclidean_difference(test_WordCounts, training_WordCounts[index])
            similarity.append([training_labels[index], euclidean_diff])
        
        # Sort list in ascending order based on euclidean difference
        similarity = sorted(similarity, key = lambda i:i[1])    
        
        # Select K nearest neighbours
        selected_Kvalues = [] 
        for i in range(K):
            selected_Kvalues.append(similarity[i])
            
        # Predicting the class of email
        result.append(get_class(selected_Kvalues))
        
        print(str(counter) + "/" + str(tsize) + " done!")
        counter += 1
        
    return result


# Main program

def main(K):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    
    # sample size of test emails to be tested. Use len(test_data) to test all test_data
    # tsize = len(test_data)
    tsize = 50
    
    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
    accuracy = accuracy_score(test_labels[:tsize], result)
    
    print("training data size\t: " + str(len(training_data)))
    print("test data size\t\t: " + str(len(test_data)))
    print("K value\t\t\t\t: " + str(K))
    print("Samples tested\t\t: " + str(tsize))
    print("% accuracy\t\t\t: " + str(accuracy * 100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))

    print(result)

main(11)


# Plotting accuracy for different test sizes

tsize = [50, 150, 300, 600, 900, 1200, 1582]
accuracy = [80.0, 82.7, 84.0, 80.7, 79.0, 76.6, 76.7]

plt.figure()
plt.ylim(0, 100)
plt.plot(tsize, accuracy)
plt.xlabel("Number of Test Samples")
plt.ylabel("% Accuracy")
plt.title("KNN Algorithm Accuracy")
plt.grid()
plt.show()






# to determine a suitable value for K

def get_k():
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    
    # sample size of test emails to be tested. Use len(test_data) to test all test_data
    tsize = 150
    
    K_accuracy = []
    for K in range(1,50, 2):
        result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
        accuracy = accuracy_score(test_labels[:tsize], result)
        K_accuracy.append([K, accuracy*100])
        print("training data size\t: " + str(len(training_data)))
        print("test data size\t\t: " + str(len(test_data)))
        print("K value\t\t\t\t: " + str(K))
        print("Samples tested\t\t: " + str(tsize))
        print("% accuracy\t\t\t: " + str(accuracy * 100))
        print("Number correct\t\t: " + str(int(accuracy * tsize)))
        print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
    K_accuracy_sorted = sorted(K_accuracy, key = lambda i:i[1])
    print(K_accuracy_sorted)
    print("MAX: " + str(max(K_accuracy_sorted, key = lambda i:i[1])))
    
    # plot
    
    K_accuracy = np.array(K_accuracy)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]
    
    plt.figure()
    plt.ylim(0, 101)
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("% Accuracy")
    plt.title("KNN Algorithm Accuracy With Different K")
    plt.grid()
    plt.show()
    
# get_k()
