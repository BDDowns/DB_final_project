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


'''
Experiment 1 analyzes the spearman rank correlation of length of movie and rating

'''
def experiment1():
    pass


def visual1():
    pass


'''
Experiment 2 analyzes the correlation of start year and budget for a movie, using the pearson correlation coefficient

'''
def experiment2():
    pass


def visual2():
    pass

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
    # read in data file
    file = open("./data/decisionTree.csv", encoding="utf8")
    x_data = []
    y_data = []
    # pull attributes from the header
    attributes = file.readline()
    attributes = attributes.strip().split(",")
    for line in file:
        # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        # take everything from before index length - 1 and store it as x values 
        x_data.append(line[:-1])
        # put the index length - 1 in the y values
        y_data.append(line[-1])

    # build a tree

    # write tree to file

'''
Experiment 5 looks at association of actors and directors by performing a market basket analysis

'''
def experiment5():
    # read in data file
    file = open("./data/assocRules_withDirector.csv", encoding="utf8")
    data = []

    # pull attributes from the header
    attributes = file.readline()
    attributes = attributes.strip().split(",")
    for line in file:
        # remove quotes with replace(), return characters with strip(), and split() into a list on commas
        line = line.replace('"', '').strip().split(',')
        data.append(line)
    print(data)

def visual5():
    pass

