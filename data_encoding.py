from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd 

def data_enc(dataset):
    '''
    Encodes categorical data with an ordinal encoder. Can extend to
    do one-hot encoding.  

    Parameters
    ==========
    dataset: pandas dataframe without nans.

    Returns
    ==========
    X_data_enc: pandas df with encoded features.
    y_data_enc: numpy array with encoded labels.

    '''
    X_data = dataset.iloc[:,:-2]
    y_data = dataset.iloc[:,-2] # label
    issue_id = dataset.iloc[:,-1] # issue id
    # encode categorical labels
    label_enc = preprocessing.LabelEncoder()
    # replace nan with a 'no label' category
    y_data.fillna(value='no_label', inplace=True)
    y_data_enc = label_enc.fit_transform(y_data.values)
    # y_label_inv_trans = label_enc.inverse_transform(y_data_enc)

    # encode features
    # print the dtypes
    # X_data.dtypes.value_counts()
    # select the subset that is dtype object
    X_data_cat = X_data.select_dtypes(include='O')
    # select the subset that is dtype float64
    X_data_num = X_data.select_dtypes(include='float64')
    # now convert X_data_cat to integers, do one-hot later
    feat_enc = OrdinalEncoder()
    X_data.fillna(value=np.nan, inplace=True)
    X_data_cat_enc = feat_enc.fit_transform(X_data_cat)
    X_data_cat_enc = pd.DataFrame(X_data_cat_enc, columns=X_data_cat.columns)
    # join arrays
#     X_data_enc = pd.concat([X_data_cat_enc, X_data_num], axis=1)
    X_data_enc = X_data_cat_enc.join(X_data_num)

    return X_data_enc, y_data_enc