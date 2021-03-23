"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Implementation of IEI optimize with random search algorithm.
"""



import numpy as np
import math
import random

# Machine learning
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(1, '')

from utils.preprocessing import *
from utils.Feature_Selection import *


class IEI_Random(object):


    """
    Optimization of IEI algorithm implementations
    """

    # Hyperparameters points of intrest
    THRESH_1_DEFAULT_VALUES = [10 * x for x in range(1, 10)]
    THRESH_2_DEFAULT_VALUES = [0.1 * x for x in range(1, 10)]
    THRESH_3_DEFAULT_VALUES = [1, 2, 3]


    def __init__(self, IEI, model,
                 performance, explainability, 
                 overall, data_path, test_size):


        '''
        Parameters
        ----------
        Input:
                1. IEI algorithm:           Iterative Ensemble of intersections.
                2. Decision tree model:     classification tree model
                3. Performance metric:      a function that gets y_predict and y_test and returns accuracy value.
                4. Explainability metric:   a function that gets the model, and the performance and return explainability value.
                5. Fusion (overall) metric: a function that gets performance score and explenability score and returns overall score
                6. Dataset path:            particular dataset path
                7. Test size:               the test_size (between 0 and 1)

        Output:
                1. Average of hyperparametr1.
                2. Average of hyperparametr2.
                3. Average of hyperparametr3.
        ----------
        ''' 

        self.IEI            = IEI
        self.model          = model
        self.performance    = performance
        self.explainability = explainability
        self.overall        = overall
        self.data_path      = data_path
        self.test_size      = test_size


    def random_search(self,
                      thresh_1_vals: list = THRESH_1_DEFAULT_VALUES,
                      thresh_2_vals: list = THRESH_2_DEFAULT_VALUES,
                      thresh_3_vals: list = THRESH_3_DEFAULT_VALUES):



        # init list for storage of the hyper-parameters
        thresh_1_list = []
        thresh_2_list = []
        thresh_3_list = []


        # init list for storage the result from random serach
        performance_list    = []
        explainability_list = []

        X, y = preprocessing(self.data_path)
        FFS  = Filter_Algorithms(X, y, test_size=self.test_size, seed=0)


        # Apply feature selection on the features according the targets
        df_Chi2   = FFS.fit_Chi2()
        df_Anova  = FFS.fit_Anova()
        df_Mutual = FFS.fit_Mutual()


        # run over random hyper parameters in the serach space
        for random_search in range(10):

            thresh_1 = random.choice(thresh_1_vals)
            thresh_2 = random.choice(thresh_2_vals)
            thresh_3 = random.choice(thresh_3_vals)


            # Storage the value of each threshold
            thresh_1_list.append(thresh_1)
            thresh_2_list.append(thresh_2)
            thresh_3_list.append(thresh_3)


            # Feature selection with IEI
            features = self.IEI(df_Chi2,
                                df_Anova,
                                df_Mutual,
                                thresh_1 = thresh_1,
                                thresh_2 = thresh_2,
                                thresh_3 = thresh_3)
    

            # Whereas threshold 2 is low (== 0.1) it may find an empty subset
            # Therefore -> Skip the point in the grid search
            if len(features) == 0:
                continue
            
            # find the metrices of this run
            performance, explainability = self.single_run(data_x   = X,
                                                          data_y   = y,
                                                          features = features)
            
            # Storage the value of the performance and explainability
            performance_list.append(performance)
            explainability_list.append(explainability)


        # Average of the threshold values
        avg_thresh_1 = int(np.average(thresh_1_list))
        avg_thresh_2 = float("{:.1f}".format(np.average(thresh_2_list)))
        avg_thresh_3 = int(np.average(thresh_3_list))
        
        # Average of the performance and explainability
        avg_performance    = float("{:.2f}".format(np.average(performance_list)))
        avg_explainability = int(np.average(explainability_list))

        print('-'*20, 'performance: {}'.format(avg_performance), 'explainability: {}'.format(avg_explainability), '-'*20)

        return [avg_thresh_1, avg_thresh_2, avg_thresh_3]


    def single_run(self,
                   data_x,
                   data_y,
                   features: list):


        '''
        Parameters
        ----------
        Input:
                1. data_x   : the data (x) for the model
                2. data_y   : the data (y) for the mode
                3. Features : the features selected by IEI

        Output:
                1. Performance.
                2. Explainability

        ----------
        '''

        # filter from the data (assume pandas' dataframe) for the needed features
        data_x = data_x[features]

        X_train, X_val, y_train, y_val = train_test_split(data_x,
                                                          data_y,
                                                          stratify     = data_y,
                                                          test_size    = self.test_size,
                                                          random_state = 0)

        # Prune the max depth of decition tree 
        max_depth = int(2*math.log2(len(features)))
        if max_depth == 0:
            max_depth = 1

        clf = self.model(max_depth=max_depth, random_state=0) 
        clf.fit(X_train, y_train)

        # Calculate performance metric
        performance    = self.performance(y_val, clf.predict(X_val))
        # Calculate explainability metric
        explainability = self.explainability(clf, performance)

        return performance, explainability