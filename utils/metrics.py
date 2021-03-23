"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Helper functions:
 1. Overall score between, explainability and performance with normalization between 0-1 (logaritmic_power, sigmoid_power).
 2. An explainability minimization (smaller is better) with additive constrains according the number of leaves and the error for the optimization.
 3. Accuracy score.
"""


from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

import numpy as np
import math

    
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def logaritmic_power(x, y):
    '''
    Parameters:
    ----------
    input:
            x: performance (scalar)
            y: explainability (scalar)
    output:
            factor: Normalized overall score
    ----------
    '''
    z      = 1-x
    l      = np.log2(y ** z)
    factor = x ** l
    return factor


def sigmoid_power(x, y):
    '''
    Parameters:
    ----------
    input:
            x: performance (scalar)
            y: explainability (scalar)
    output:
            factor: Normalized overall score
    ----------
    '''
    sigmoid  = 1/(1 + math.exp(-y))
    factor   = x ** sigmoid
    return factor


def explainability_metric(clf, x):
    '''
    Parameters:
    ----------
    input:
            x:   performance
            clf: object of decision tree 
    
    output:
            minimize: explainable (scalar)
    ----------       
    '''
    size_leaf = clf.tree_.n_leaves
    size_node = len([z for z in clf.tree_.feature  if z != _tree.TREE_UNDEFINED])
    _lambda   = 1
    error     = (1.0 - x)
    minimize  = error + _lambda * size_node

    return minimize