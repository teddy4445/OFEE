"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Implementation of IEI optimize with grid search algorithm.
"""


import sys
import math
import numpy as np
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split

from utils.preprocessing import *
from utils.Feature_Selection import *

sys.path.insert(1, '')


class IEI_Grid(object):
 

    # Hyperparameters points of intrest for grid search
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
                1. The best features according explainability and performance
                2. Best hyperparametr1.
                3. Best hyperparametr2.
                4. Best hyperparametr3.
        ----------
        ''' 


        self.IEI            = IEI
        self.model          = model
        self.performance    = performance
        self.explainability = explainability
        self.overall        = overall
        self.data_path      = data_path
        self.test_size      = test_size


    def grid_search(self,
                    thresh_1_vals: list = THRESH_1_DEFAULT_VALUES,
                    thresh_2_vals: list = THRESH_2_DEFAULT_VALUES,
                    thresh_3_vals: list = THRESH_3_DEFAULT_VALUES):

        
        # init best option for meta-data
        best_thresh_1       = None
        best_thresh_2       = None
        best_thresh_3       = None
        best_features       = None
        best_performance    = 0.0
        best_explainability = 0.0
        best_score          = 0.0


        X, y = preprocessing(self.data_path)
        FFS  = Filter_Algorithms(X, y, test_size=self.test_size, seed=0)


        # Apply feature selection on the features according the targets
        df_Chi2   = FFS.fit_Chi2()
        df_Anova  = FFS.fit_Anova()
        df_Mutual = FFS.fit_Mutual()
		
		
		# TODO: replace with skelearn GridSearch design 
        # run over the whole hyperparameters space
        for thresh_1 in thresh_1_vals:
            for thresh_2 in thresh_2_vals:
                for thresh_3 in thresh_3_vals:

                    # Feature selection with IEI
                    features = self.IEI(df_Chi2,
                                        df_Anova,
                                        df_Mutual,
                                        thresh_1 = thresh_1,
                                        thresh_2 = thresh_2,
                                        thresh_3 = thresh_3)

                    
                    # Whereas hyperparameter2 is low (== 0.1) it may find an empty subset.
                    # Therefore -> Skip the point in the grid search.
                    if len(features) == 0:
                        continue
                    
                    # Find the metrices of this run
                    performance, explainability = self.single_run(data_x   = X,
                                                                  data_y   = y,
                                                                  features = features)

                    # Check if better overll
                    model_score = self.overall(performance, explainability)
                    
                    # Boolean for best selection method
                    if model_score          > best_score:
                        best_thresh_1       = thresh_1
                        best_thresh_2       = thresh_2
                        best_thresh_3       = thresh_3
                        best_explainability = explainability
                        best_performance    = performance
                        best_features       = features
                        best_score          = model_score



        best_explainability = int(best_explainability)
        best_performance    = float("{:.2f}".format(best_performance))

        print('-'*20, 'performance: {}'.format(best_performance), 'explainability: {}'.format(best_explainability), '-'*20)

        return [best_features, best_thresh_1, best_thresh_2, best_thresh_3]



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