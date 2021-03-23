"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Calculation explainability in Adabbost algorithm.
"""



import sys
import math
import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from utils.preprocessing import *


sys.path.insert(1, '')


class Adaboost:
    def __init__(self, model, performance, explainability, 
                 data_path, test_size):

 

        '''
        Parameters
        ----------
        Input:
                1. Model:                   Adaboost model.
                2. Performance metric:      a function that gets y_predict and y_test and returns accuracy value.
                3. Explainability metric:   a function that gets the model, and the performance and return explainability value.
                4. Dataset path:            particular dataset path
                5. Test size:               the test_size (between 0 and 1)

        Output:
                1. Explainability.
                2. Performance.
        ----------
        ''' 
 
        self.model          = model
        self.performance    = performance
        self.explainability = explainability
        self.data_path      = data_path
        self.test_size      = test_size


    def fit(self):

        # Define the features and the targets with preprocessing
        X, y = preprocessing(self.data_path)
        
        # Split the data with all the possible features
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=self.test_size, random_state=0)

        max_depth = int(2*math.log2(X.shape[1]))
        if max_depth == 0:
            max_depth = 1

        # Run Adaboost algorithm
        clf       = AdaBoostClassifier(random_state=0, base_estimator=self.model(max_depth=max_depth))
        clf.fit(X_train, y_train)
        clf       = clf.estimators_[-1]
        y_pred    = clf.predict(X_val)

        # Calculate performance metric
        performance    = self.performance(y_val, clf.predict(X_val))
        performance    = float("{:.2f}".format(performance))
        # Calculate explainability metric
        explainability = int(self.explainability(clf, performance))

        print('-'*20, 'performance: {}'.format(performance), 'explainability: {}'.format(explainability), '-'*20)


        return explainability, performance