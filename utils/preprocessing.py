"""
 Authors: Theodore Lazebnik, Roni Reznik Leor

 utility function for reading and preapering the data for IEI and IEIO.
"""


import os
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.io import loadmat  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocessing(data_path):

    np.random.seed(0)
    path = data_path
    data_type = path.split(os.sep)[-1].split(".")[-1]
    data_name = path.split(os.sep)[-1].split(".")[0]

    # Load arff format with python
    if data_type == "arff":
        data = arff.loadarff(path)
        df   = pd.DataFrame(data[0])

    # Load csv format with python
    elif data_type == "csv":
        df = pd.read_csv(path)


    # Dealing with missing values
    col_miss_val = []
    for i in range(df.shape[1]):
        if df.isna().sum()[i] != 0:
            col_miss_val.append(df.columns.to_list()[i])
        else:
            continue

    # Calcualte the median of particular feature with missing values
    median = []
    for i in col_miss_val:
        median.append(df[i].median())

    # Fill the missing values in particular features with median 
    for i, k in zip(col_miss_val, range(len(col_miss_val))):
        df[i] = df[i].fillna(median[k])


    # Encoding categories features to labels encoding
    cat_index = []
    for i in range(df.shape[1]):
        if df.iloc[:,i].dtype == object:
            cat_index.append(i)

    for i in cat_index:
        encode = LabelEncoder()
        df[df.iloc[:,i].name] = encode.fit_transform(df[df.iloc[:,i].name])


    # Normalize the data with min max scale
    scaler    = MinMaxScaler()
    df_scaled = scaler.fit_transform(df.iloc[:,:-1].values)
    df        = pd.DataFrame(data=np.concatenate((df_scaled, df.iloc[:,-1].values.reshape((-1,1))), axis=1), columns=df.columns.to_list())


    # Define the features and the targets
    X, y = df.iloc[:,:-1], df.iloc[:,-1]

    return X, y