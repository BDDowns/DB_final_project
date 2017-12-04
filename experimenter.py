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
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr




'''
Experiment 1 analyzes the spearman rank correlation of length of movie and rating

'''
# def experiment1():
#     # read in data file
#     file1 = open("./data/movie_length.csv", encoding="utf8")
#     file2 = open("./data/imdb_score.csv", encoding="utf8")
#     x = file1
#     y = file2
#     for line in file1:
#         # remove quotes with replace(), return characters with strip(), and split() into a list on commas
#         line = line.replace('"', '').strip().split(',')
#         x.append(line)
#         file1.close()
#     for line in file2:
#         # remove quotes with replace(), return characters with strip(), and split() into a list on commas
#         line = line.replace('"', '').strip().split(',')
#         x.append(line)
#         x = np.array(data)
#         y = np.array(data)
#     #show spearman rank correlation coefficient as first input, and then p-value as second output
#     print(spearmanr(x,y))
#     #print out scatter plot
#     mpl.scatter(x, y, s=area, c=colors, alpha=0.5)
#     mpl.title('Movie Length compared to Critic Score')
#     mpl.xlabel('Movie Length')
#     mpl.ylabel('Movie Score')
#     mpl.show()


# '''
# Experiment 2 analyzes the correlation of start year and budget for a movie, using the pearson correlation coefficient

# '''
# def experiment2():
#  # read in data file
#     file1 = open("./data/movie_length.csv", encoding="utf8")
#     file2 = open("./data/imdb_score.csv", encoding="utf8")
#     x = file1
#     y = file2
#     for line in file1:
#     # remove quotes with replace(), return characters with strip(), and split() into a list on commas
#     line = line.replace('"', '').strip().split(',')
#     x.append(line)
#     file1.close()
#     for line in file2:
#     # remove quotes with replace(), return characters with strip(), and split() into a list on commas
#     line = line.replace('"', '').strip().split(',')
#     x.append(line)
#     x = np.array(data)
#     y = np.array(data)
#     #show pearson correlation coefficient as first input, and then p-value as second output
#     print(pearsonr(x,y))
#     #print out scatter plot
#     mpl.scatter(x, y, s=area, c=colors, alpha=0.5)
#     mpl.title('Budget Over Time')
#     mpl.xlabel('Release Year')
#     mpl.ylabel('Budget')
#     mpl.show()
'''
Experiment 3 performs a regression analysis of the rating and runtime of a television series 

'''
def experiment3():
    pass


def visual3():
    pass

'''
Experiment 4 uses a regression decision tree to predict imdb_score of a film given 7 features of the data
It operates as a standalone function

'''
def experiment4():
    # read in as datafile with pandas
    df = pd.read_csv("./data/decisionTree.csv")
    # drop rows with missing values for error correction
    df = df.dropna()

    df_norm = preprocessing.MinMaxScaler().fit(df[['duration','net_revenue','imdbScore']])
    df[['duration','net_revenue','imdbScore']] = df_norm.transform(df[['duration','net_revenue','imdbScore']])

    # split the data by selecting columns in csv for features and class/value
    X, y = df.iloc[1:,:-1], df.iloc[1:,-1]
    # pd.get_dummies(x) makes matrix representation for categorical data to eliminate ordinality imposed by assigning random numerical values
    X_encoded = pd.get_dummies(X)

    # split a train and test set for cross_validated sampling
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

    # create a regression tree object, with maximum tree depth set to 5. this is tunable
    regTree = DecisionTreeRegressor(max_depth=5)
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
It pits a classification decision tree with a multilayer perceptron classification neural network
using identical attribute sets 

'''
def experiment5():
    decisionTreeClassifier()
    neuralNetworkClassifier()

def decisionTreeClassifier():
    # create a decision tree
    df = pd.read_csv("./data/decisionTree.csv")
    df = df.dropna()

    df_norm = preprocessing.MinMaxScaler().fit(df[['duration','net_revenue','imdbScore']])
    df[['duration','net_revenue','imdbScore']] = df_norm.transform(df[['duration','net_revenue','imdbScore']])

    X, y = df.iloc[1:,[0,1,2,3,4,6,7]], df.iloc[1:,[5]]
    X_encoded = pd.get_dummies(X)
    

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)


    classTree = DecisionTreeClassifier(max_depth=5)
    classTree.fit(X_train, y_train)

    y_predictions = classTree.predict(X_test)

    print('Accuracy: {0:.3f}'.format(sk.metrics.accuracy_score(y_test, y_predictions)), "\n")

    export_graphviz(classTree, out_file="./results/classifierTreeD5.dot")

def neuralNetworkClassifier():
    # create a neural network
    df = pd.read_csv("./data/decisionTree.csv")
    df = df.dropna()
    # the large range of revenue values was causing gradient explosion and killing error
    # using a data normalization between 0 and 1 to gain more accuracy and compete with the decision tree
    # however, we will normalize all numerical attributes between 0 and 1
    df_norm = preprocessing.MinMaxScaler().fit(df[['duration','net_revenue','imdbScore']])
    df[['duration','net_revenue','imdbScore']] = df_norm.transform(df[['duration','net_revenue','imdbScore']])

    X, y = df.iloc[1:,[0,1,2,3,4,6,7]], df.iloc[1:,[5]]
    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

    mlpR = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=1)

    mlpR.fit(X_train, y_train)

    y_predictions = mlpR.predict(X_test)
    
    print('Accuracy: {0:.3f}'.format(sk.metrics.accuracy_score(y_test, y_predictions)), "\n")

