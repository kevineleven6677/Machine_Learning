# CS 675 Course Project
# Team Member: Yarou Ding, Kevin Chen


import sys
import gzip
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



###The function of getting feature data from file as a matrix with a row per data instance

def getFeatureData(featureFile):
    x=[]
    dFile = gzip.open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x


####The function of getting label data from file as a dictionary with key as data instance index
### and value as the class index

def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict


#### The function of computing chi-square of a single feature

def chiSquare(feature, label):
    assert len(feature) == len(label)
    
    a1 = b1 = c1 = d1 = e1 = f1 = 0
    for i in range(len(feature)):
        if feature[i] == 0 and label[i] == 0:
            a1 += 1
        elif feature[i] == 0 and label[i] == 1:
            b1 += 1
        elif feature[i] == 1 and label[i] == 0:
            c1 += 1
        elif feature[i] == 1 and label[i] == 1:
            d1 += 1
        elif feature[i] == 2 and label[i] == 0:
            e1 += 1
        elif feature[i] == 2 and label[i] == 1:
            f1 += 1
        # else:
        #     print("chi-square error")
        #     return
    
    r = a1 + b1 + c1 + d1 + e1 + f1
    a2 = (a1 + b1) * (a1 + c1 + e1) / r
    b2 = (a1 + b1) * (b1 + d1 + f1) / r
    c2 = (c1 + d1) * (a1 + c1 + e1) / r
    d2 = (c1 + d1) * (b1 + d1 + f1) / r
    e2 = (e1 + f1) * (a1 + c1 + e1) / r
    f2 = (e1 + f1) * (b1 + d1 + f1) / r 
    
    # Avoid division by zero
    a = 0 if a2 == 0 else (a1 - a2) ** 2 / a2
    b = 0 if b2 == 0 else (b1 - b2) ** 2 / b2
    c = 0 if c2 == 0 else (c1 - c2) ** 2 / c2
    d = 0 if d2 == 0 else (d1 - d2) ** 2 / d2
    e = 0 if e2 == 0 else (e1 - e2) ** 2 / e2
    f = 0 if f2 == 0 else (f1 - f2) ** 2 / f2
    
    return a + b + c + d + e + f


##############################  Main function  ####################################
### 1. reading file

def main():
    print("Running result might takes around 10 mins. Please wait , Thank you for your patience.")

    ########Read data
    # Get the file names from the command line
    trainingData = sys.argv[1]
    trainingLabels = sys.argv[2]
    testData = sys.argv[3]
    
    ######## Local test for reading data
    
    trainingData = getFeatureData(r"C:\Users\wendy\OneDrive\data\course project\traindata.gz")
    trainingLabels = getFeatureData(r"C:\Users\wendy\OneDrive\data\course project\trueclass.txt")
    testData = getFeatureData(r"C:\Users\wendy\OneDrive\data\course project\testdata.gz")


    # x_train_original is the training data matrix (row: samples, column: features)
    x_train_original = getFeatureData(trainingData)

    # x_train is the dictionary (key=row number:  value=class)
    x_label = getLabelData(trainingLabels)

    # y is the label array
    y = []
    for key, value in x_label.items():
        y.append(value)
        
    # x_test_original is the test data matrix (row: samples, column: features)
    x_test_original = getFeatureData(testData)

    # n is the number of samples of training data
    n = len(x_train_original)
    # m is the number of features of training data
    m = len(x_train_original[0])


   
    ### 2. Feature selection by computing chi-square 
    
    # chi_squares is dictionary which stores the pairs of 
    # column number and corresponding chi-square value
    # chi_squares = (key=column number: value=chi-square value)
    chi_squares = {}
    # For each column, we compute its chi-square and then 
    # store the chi-square value into the chi_squares dictionary
    for i in range(m):
        chiSq = chiSquare([row[i] for row in x_train_original], y)
        chi_squares[i] = chiSq

    # The greater the chi-square value is, the more important the feature is to the problem.
    sorted_features = sorted(chi_squares, key=chi_squares.get, reverse=True)
    # We only pick the top 15 features which have the biggest chi-square values in decreasing order
    selected_features = sorted_features[:15]

    # Now, we created the new training dataset
    X = []
    for i in range(n):
        X.append([])
        for j in range(15):
            X[i].append(x_train_original[i][selected_features[j]])


    
    ####   3. Train and compute accuracy 
    # Randomly select 75% data as training data and 25% data as validation data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Train by using support 
    clf = SVC(kernel="poly")
    clf.fit(x_train, y_train)
    print("The accuracy of validation data is", str(round(clf.score(x_test, y_test)*100, 2)) + "%")

    ####  4. Predict the test data  
    # Prepare test data
    X_test = []
    for i in range(len(x_test_original)):
        X_test.append([])
        for j in range(15):
            X_test[i].append(x_test_original[i][selected_features[j]])

    # Predict test data
    labels = clf.predict(X_test)
    for i in range(len(x_test_original)):
        print(labels[i], i)

    # Print the total number and column number of the selected features
    print("\nOnly 15 features are used in our training!")
    print("Their column numbers are", selected_features)

    # Print the selected features
    print("\n15 selected features are as follows,")
    for row in range(len(X)):
        for col in X[row]:
            print(col, end=" ")
        print()


############################Running the Main function###################################

main()




