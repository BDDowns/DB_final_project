'''
Experimenter.py controls all experiments for the datascience portion of Databases.
Each experiment is layed out such that it contains the actual code to perform analysis on the data
and a seperate function to output a visual representation of said exeriment.

@authors: Bryan Downs, Brett Shelley, Brendan Tracey

'''

import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib as mpl
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr




'''
Experiment 1 analyzes the spearman rank correlation of length of movie and rating

'''
def experiment1():
    #read in files and adjust them with panda
    file1 = pd.read_csv("./data/movie_length.csv")
    file2 = pd.read_csv("./data/imdb_score.csv")
    x = file1
    y = file2
    colors = (0,0,0)
    area = np.pi*3
    #show the spearman rank correlation
    print(spearmanr(x,y))
    #display the scatterplot
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Movie Length compared to Critic Score')
    plt.xlabel('Movie Length')
    plt.ylabel('Movie Score')
    plt.show()
    #save the 
    fig = plt.figure()
    fig.savefig('Results1.png')


# '''
# Experiment 2 analyzes the correlation of start year and budget for a movie, using the pearson correlation coefficient

# '''
 def experiment2():
    file1 = pd.read_csv("./data/release_year.csv")
    file2 = pd.read_csv("./data/budget.csv")
    x = file1
    y = file2
    colors = (0,0,0)
    area = np.pi*3
    print(spearmanr(x,y))
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title('Movie Budget Over Time')
    plt.xlabel('Release Year')
    plt.ylabel('Movie Budget')
    plt.show()
    fig = plt.figure()
    fig.savefig('Results2.png')
'''
Experiment 3 performs a regression analysis of the rating and runtime of a television series 

'''
def experiment3():
    pass


def visual3():
    pass

'''
Experiment 4 uses a Decision Tree Regression Algorithm to predict imdb_score of a film given the following features of the data:
Director, Actor 1 Name, Actor 2 Name, Actor 3 Name, Duration, Content Rating, Net Revenue

@return prints mean squared error to console and outputs a graphic of the completed tree
'''
def experiment4():
    # read in as datafile with pandas
    df = pd.read_csv("./data/decisionTree.csv")
    # drop rows with missing values for error correction
    df = df.dropna()

    # normalize real values to values between 0 and 1
    df_norm = preprocessing.MinMaxScaler().fit(df[['duration','net_revenue']])
    df[['duration','net_revenue']] = df_norm.transform(df[['duration','net_revenue']])

    # split the data by selecting columns in csv for features and class/value
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    # pd.get_dummies(x) makes matrix representation for categorical data to eliminate ordinality imposed by assigning random numerical values
    X_encoded = pd.get_dummies(X)

    # split a train and test set for cross_validated sampling
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

    # create a regression tree object, with maximum tree depth set to 5. this is tunable
    regTree = DecisionTreeRegressor(max_depth=3)
    # train the tree
    regTree.fit(X_train,y_train)
    # create list of predicted values based on test set
    y_hat = regTree.predict(X_test)

    # report mean squared error as the square of (ytest - yhat)
    print('MSE: {0:.3f}'.format(sk.metrics.mean_squared_error(y_test,y_hat)), "\n")

    # output visualization tool for completed tree
    export_graphviz(regTree, out_file="./results/regressionTreeD3.dot")


'''
Experiment 5 looks at the performance difference in machine learning classification problems
It pits a classification decision tree, multilayer perceptron classification neural network, and naive bayes gaussian classifier
against eachother using identical attribute sets to predict the rating of a movie using the following features:
Director, Actor 1 Name, Actor 2 Name, Actor 3 Name, Duraction, Net Revenue and IMBDScore.

@return prints comparative error of all three algorithms to console
'''
def experiment5():
    # grab data file
    df = pd.read_csv("./data/decisionTree.csv")
    df = df.dropna()
    # the large range of revenue values alone was killing accuracy (accuracy < 0.01%).
    # As a result we implemented data normalization between 0 and 1 to gain more accuracy and compete with
    # the decision tree
    df_norm = preprocessing.MinMaxScaler().fit(df[['duration','net_revenue','imdbScore']])
    df[['duration','net_revenue','imdbScore']] = df_norm.transform(df[['duration','net_revenue','imdbScore']])

    # split data into input / output and then cast categorical data
    X, y = df.iloc[:,[0,1,2,3,4,6,7]], df.iloc[:,[5]]
    X_encoded = pd.get_dummies(X)
    
    # split into train / test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

    # run data over 3 classifiers
    dtc_accuracy = decisionTreeClassifier(X_train, X_test, y_train, y_test)
    mlp_accuracy = neuralNetworkClassifier(X_train, X_test, y_train, y_test)
    nbc_accuracy = naiveBayesClassifier(X_train, X_test, y_train, y_test)

    # output results
    print('Experiment 5 results: \n')
    print('Decision Tree Classifier Accuracy: {0:.3f}%'.format(dtc_accuracy), "\n")
    print('Multi-layer Perceptron Classifier Accuracy: {0:.3f}%'.format(mlp_accuracy), "\n")
    print('Naive Bayes Gaussian Classifier Accuracy: {0:.3f}%'.format(nbc_accuracy), "\n")


'''
DecisionTreeClassifier Builds, Trains and Tests a DecisionTreeClassifier

@param X_train, X_test, y_train, y_test input training, test, output training test examples, respectively.
@return error as percent classified correctly
'''
def decisionTreeClassifier(X_train, X_test, y_train, y_test):
    # create the tree
    classTree = DecisionTreeClassifier(max_depth=3)
    # train and test
    y_predictions = classTree.fit(X_train, y_train).predict(X_test)
    # output pretty graphics
    export_graphviz(classTree, out_file="./results/classifierTreeD3.dot")
    # return accuracy
    return sk.metrics.accuracy_score(y_test, y_predictions)

'''
NeuralNetworkClassifier Builds, Trains and Tests a Multilayer Perceptron NN

@param X_train, X_test, y_train, y_test input training, test, output training test examples, respectively.
@return error as percent classified correctly
'''
def neuralNetworkClassifier(X_train, X_test, y_train, y_test):
    # create the neural network
    mlpC = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
    # train and test
    y_predictions = mlpC.fit(X_train, y_train).predict(X_test)
    # return results as accuracy
    return sk.metrics.accuracy_score(y_test, y_predictions)

'''
NaiveBayesClassifier Builds, Trains and Tests a Gaussian Naive Bayes Classifier

@param X_train, X_test, y_train, y_test input training, test, output training test examples, respectively.
@return error as percent classified correctly
'''
def naiveBayesClassifier(X_train, X_test, y_train, y_test):
    # create the bayes net
    nbc = GaussianNB()
    # train and test
    y_predictions = nbc.fit(X_train, y_train).predict(X_test)
    # return results as accuracy
    return sk.metrics.accuracy_score(y_test, y_predictions)
