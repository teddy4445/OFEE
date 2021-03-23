import sys

sys.path.insert(1, '/')
from OIEI.IEI_RandomSearch import *
from OIEI.IEI_GridSearch import *
from OIEI.Adaboost import * 

from OIEI.IEI import *

from utils.metrics import *
from utils.graph import *
from sklearn.tree import DecisionTreeClassifier


data_path = '/explanatory_ml_research/Datasets/semeion.arff'
Adaboost(model=DecisionTreeClassifier,performance=get_score, explainability=explainability_metric,
         data_path=data_path, test_size=0.3).fit()


data_path = '/explanatory_ml_research/Datasets/semeion.arff'
features, thresh_1, thresh_2, thresh_3  = IEI_Grid(IEI            = IEI,
                                                   model          = DecisionTreeClassifier,
                                                   performance    = get_score,
                                                   explainability = explainability_metric,
                                                   overall        = logaritmic_power,
                                                   data_path      = data_path,
                                                   test_size      = 0.3).grid_search()


data_path = '/explanatory_ml_research/Datasets/semeion.arff'
avg_thresh_1, avg_thresh_2, avg_thresh_3  = IEI_Random(IEI            = IEI,
                                                       model          = DecisionTreeClassifier,
                                                       performance    = get_score,
                                                       explainability = explainability_metric,
                                                       overall        = logaritmic_power,
                                                       data_path      = data_path,
                                                       test_size      = 0.3).random_search()


data_path = '/explanatory_ml_research/Datasets/semeion.arff'
IEI_plot(IEI=IEI, thresh_1=5, thresh_2=0.9, thresh_3=3, data_path=data_path, test_size=0.3).plot()