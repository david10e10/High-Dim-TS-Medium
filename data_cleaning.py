import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold



def cleaner(file):
    '''
    Cleans a parquet file by dropping all columns with nulls or zeros.

    Parameters
    ==========
    file: csv file with features and labels to clean.

    Returns
    ==========
    dataset: cleaned dataset.

    '''
    df_all = pd.read_csv(file)

    # remove cols with all nans
    df_all.dropna(axis=1, how='all',inplace=True)

    # remove cols with all 0s
    df_non_zero = df_all.loc[:,(df_all != 0).any(axis=0)]

    # remove cols using variance threshold 
    # selector = VarianceThreshold(threshold=0.2)
    # df_varianced = selector.fit_transform(df_all)

    dataset = df_non_zero

    return dataset
