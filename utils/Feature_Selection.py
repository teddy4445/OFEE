"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Feature Selection module of chi2, anova, and mutual information.
 The main object is to insert X, y, and output an dataframe with features and their scores.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif, SelectKBest


class Filter_Algorithms(object):
        
    def __init__(self, X, y, test_size, seed=0):
        
        """
        Parameters
        ----------
        input: 
                X: array-like {n_samples, n_features}
                    Training instances to compute the feature importance scores
                y: array-like {n_samples}
                    Training labels

        output:
                R: Ranked features according particular algorithm
        -------
        """    
        
        self.X         = X          # Feature values
        self.y         = y          # Target values
        self.seed      = seed       # Fixed seed
        self.test_size = test_size  # Split for train and test


    def fit_Chi2(self):
        scores_Chi2     = []           
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, stratify=self.y, test_size=self.test_size, random_state=self.seed)        
        X_train         = pd.DataFrame(data=X_train, columns=self.X.columns)        
        scores, pvalues = chi2(X_train, y_train)                  
        for i in range(X_train.shape[1]):
            scores_Chi2.append((scores[i], X_train.columns[i]))
            df_Chi2    = pd.DataFrame(data=scores_Chi2, columns=('score', 'feature'))  
            blankIndex = [''] * len(df_Chi2)
            df_Chi2.index = blankIndex
            df_Chi2    = df_Chi2.sort_values(by='score', ascending=False)
            
        return df_Chi2


    def fit_Anova(self):        
        scores_Anova    = []           
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, stratify=self.y, test_size=self.test_size, random_state=self.seed)        
        X_train         = pd.DataFrame(data=X_train, columns=self.X.columns)        
        scores, pvalues = f_classif(X_train, y_train)                  
        for i in range(X_train.shape[1]):
            scores_Anova.append((scores[i], X_train.columns[i]))
            df_Anova = pd.DataFrame(data=scores_Anova, columns=('score', 'feature'))  
            blankIndex=[''] * len(df_Anova)
            df_Anova.index = blankIndex
            df_Anova = df_Anova.sort_values(by='score', ascending=False)
            
        return df_Anova


    def fit_Mutual(self):        
        scores_Mutual = []           
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, stratify=self.y, test_size=self.test_size, random_state=self.seed)        
        X_train = pd.DataFrame(data=X_train, columns=self.X.columns)        
        scores  = mutual_info_classif(np.array(X_train), np.array(y_train))                  
        for i in range(X_train.shape[1]):
            scores_Mutual.append((scores[i], X_train.columns[i]))
            df_Mutual = pd.DataFrame(data=scores_Mutual, columns=('score', 'feature'))  
            blankIndex=[''] * len(df_Mutual)
            df_Mutual.index = blankIndex
            df_Mutual = df_Mutual.sort_values(by='score', ascending=False)
            
        return df_Mutual