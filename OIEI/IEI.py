"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Implementation of IEI algorithm.
"""

import numpy as np


def IEI(df_Chi2, df_Anova, df_Mutual, thresh_1, thresh_2, thresh_3=3, plot=False):
    
    '''
    Parameters
    ----------
     Input:
            1. Ranked features from 3 feature selection algorithms (df_Chi2, df_Anova, df_Mutual)
            2. Hyperparmeter1 (thresh_1) - The final subset of features you desire
            3. Hyperparmeter2 (thresh_2) - The iterative cut of ranked features
            4. Hyperparmeter3 (thresh_3) - Aggregation of features in each iteration

     Output:
            1. The final subset if plot == False.
            2. List of the number of features in each iteration if plot == True.
    ----------
    '''  
    
    chi2_list       = list(df_Chi2['feature'])
    Anova_list      = list(df_Anova['feature'])
    Mutual_list     = list(df_Mutual['feature'])


    intersections   = []
    x_range         = []


    rank_chi2   = df_Chi2.set_index('feature').T.to_dict('list')
    rank_anova  = df_Anova.set_index('feature').T.to_dict('list')
    rank_mutual = df_Mutual.set_index('feature').T.to_dict('list')


    # Hyperparmeter3 - union
    if thresh_3 == 1:
        def func(a, b, c): 
            return list(set(a).union(b).union(c))

    # Hyperparmeter3 - occurrence of 2 feature at least 
    elif thresh_3 == 2:
        def func(a, b, c): 
            x = set(a).intersection(b)
            y = set(a).intersection(c)
            z = set(b).intersection(c)
            return list(set(x).union(y).union(z))

    # Hyperparmeter3 - intersection
    elif thresh_3 == 3:
        def func(lst1, lst2, lst3): 
            return [item for item in lst1 if item in lst2 and item in lst3]


    # Arbitrary start for while loop 
    condtion = [x for x in range(20000)]
    # Hyperparmeter1
    while len(condtion) >= thresh_1:
        
        # Sorting the features list according the original rank
        chi2_list    = sorted(chi2_list,   key=rank_chi2.get,   reverse=True)
        Anova_list   = sorted(Anova_list,  key=rank_anova.get,  reverse=True)
        Mutual_list  = sorted(Mutual_list, key=rank_mutual.get, reverse=True)

        # Iterative cut of the features according to Hyperparmeter2
        chi2_list    = chi2_list[0:int(np.floor(len(chi2_list) * thresh_2))]
        Anova_list   = Anova_list[0:int(np.floor(len(Anova_list) * thresh_2))]
        Mutual_list  = Mutual_list[0:int(np.floor(len(Mutual_list) * thresh_2))]

        # Aggregation of the features according to Hyperparmeter3
        condtion     = func(chi2_list, Anova_list, Mutual_list)
        
        intersections.append(condtion)
        x_range.append(len(condtion))

    if plot:
        return x_range

    else:
        return intersections[-1]