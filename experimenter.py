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

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.cross_validation import train_test_split



'''
Experiment 1 analyzes the spearman rank correlation of length of movie and rating

'''
def experiment1():
    # read in data file
        file1 = open("./data/movie_length.csv", encoding="utf8")
        file2 = open("./data/imdb_score.csv", encoding="utf8")
        x = file1
        y = file2
        for line in file1:
            # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        x.append(line)
        file1.close()
        for line in file2:
            # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        x.append(line)
        x = np.array(data)
        y = np.array(data)
    #show spearman rank correlation coefficient as first input, and then p-value as second output
    print(spearmanr(x,y))
    #print out scatter plot
    mpl.scatter(x, y, s=area, c=colors, alpha=0.5)
    mpl.title('Movie Length compared to Critic Score')
    mpl.xlabel('Movie Length')
    mpl.ylabel('Movie Score')
    mpl.show()


'''
Experiment 2 analyzes the correlation of start year and budget for a movie, using the pearson correlation coefficient

'''
def experiment2():
 # read in data file
        file1 = open("./data/movie_length.csv", encoding="utf8")
        file2 = open("./data/imdb_score.csv", encoding="utf8")
        x = file1
        y = file2
        for line in file1:
            # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        x.append(line)
        file1.close()
        for line in file2:
            # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        x.append(line)
        x = np.array(data)
        y = np.array(data)
    #show pearson correlation coefficient as first input, and then p-value as second output
    print(pearsonr(x,y))
    #print out scatter plot
    mpl.scatter(x, y, s=area, c=colors, alpha=0.5)
    mpl.title('Budget Over Time')
    mpl.xlabel('Release Year')
    mpl.ylabel('Budget')
    mpl.show()
'''
Experiment 3 performs a regression analysis of the rating and runtime of a television series 

'''
def experiment3():
    pass


def visual3():
    pass

'''
Experiment 4 uses a decision tree to predict imdb_score of a film given 7 features of the data

'''
def experiment4():
    df = pd.read_csv("./data/decisionTree.csv")
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    X_encoded = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=1)

    regTree = DecisionTreeRegressor(max_depth=3)
    regTree.fit(X_train,y_train)

    y_hat = regTree.predict(X_test)

    print('MSE: {0:.3f}'.format(sk.metrics.mean_squared_error(y_test,y_hat)), "\n")

    export_graphviz(regTree, out_file="./results/regressionTreeD3.dot")


'''
Experiment 5 looks at association of actors and directors by performing a market basket analysis

'''
def experiment5():
    # read in data file
    file = open("./data/assocRules_withDirector.csv", encoding="utf8")
    data = []

    # pull feature_names from the header
    feature_names = file.readline()
    feature_names = feature_names.strip().split(",")
    feature_names = np.array(feature_names)
    for line in file:
        # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        if all(x for x in line):
            data.append(line)
    file.close()
    data = np.array(data)
    

def visual5():
    pass

experiment4()