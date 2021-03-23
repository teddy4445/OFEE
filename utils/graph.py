"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 Visualization of IEI convergence constrained to thresh_1, with respect to thresh_2 and thresh_3.
"""




import sys
import os
sys.path.insert(1, '/')

from IEI.IEI import *
from utils.preprocessing import *
from utils.Feature_Selection import *

import matplotlib.pyplot as plt


class IEI_plot(object):

    def __init__(self, IEI, thresh_1, thresh_2, thresh_3, data_path, test_size):

        # Call IEI algorithm
        self.IEI       = IEI
        # Select thresholds
        self.thresh_1  = thresh_1
        self.thresh_2  = thresh_2
        self.thresh_3  = thresh_3
        # The data you desire to explore
        self.data_path = data_path
        self.test_size = test_size

    def plot(self):

        # Call preprocessing of the data
        X, y = preprocessing(self.data_path)
        # Call to feature selection algorithms
        FFS  = Filter_Algorithms(X, y, test_size=self.test_size, seed=0)


        # Apply feature selection on the features according the targets
        df_Chi2   = FFS.fit_Chi2()
        df_Anova  = FFS.fit_Anova()
        df_Mutual = FFS.fit_Mutual()

        
        # Return x-range of features during the iterations
        features = self.IEI(df_Chi2,
                            df_Anova,
                            df_Mutual,
                            thresh_1 = self.thresh_1,
                            thresh_2 = self.thresh_2,
                            thresh_3 = self.thresh_3,
                            plot     = True)


        os.chdir('/expirements')
        # making dir of results
        if not os.path.exists('results'):
            os.mkdir('results')
            os.chdir('results')
        else:
            os.chdir('results')



        # Precentage of features reducing without IEI
        cut_precentage = [self.thresh_2 ** i for i in range(len(features))]


        data_name = self.data_path.split(os.sep)[-1].split('.')[0]
        plt.style.use('seaborn-paper')
        fig, ax1 = plt.subplots(figsize=(10,8))


        ax2 = ax1.twinx()
        ax1.plot(features, '--', color='blue')
        ax1.plot(features, 'o',  color='blue')
        ax2.plot(cut_precentage, '--', color='green')
        ax2.plot(cut_precentage, 'o',  color='green')


        ax1.set_xlabel('Number of iterations')
        ax1.set_ylabel('Feature Space', color='b')
        ax2.set_ylabel('Iterative Cut', color='g')
        x_axis = [i for i in range(0,len(features),1)]
        plt.xticks((x_axis))

        plt.title(f'$Homogenous\ Ensemble\ on\ {data_name}\ with\ thresh_1={self.thresh_1}, \
                  \ thresh_2={self.thresh_2}\%,\ and\ thresh_3={self.thresh_3}$')
        plt.savefig(f'IEI_{data_name}_{self.thresh_1}_{self.thresh_2}%_{self.thresh_3}.jpg')
        plt.show()