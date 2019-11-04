'''
Implements various dimensionality reduction techiques
for later classification and for data exploration

'''

import os
import numpy as np 
import pandas as pd

from sklearn.decomposition import KernelPCA, PCA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import preprocessing

import matplotlib.pyplot as plt


def dim_exploration(X_train):
    '''
    converts original feature data to PCA and plots the explained variance 
    cumulative sum. This gives us an idea of how many components are needed
    for other non-linear dim reduction methods such as kpca

    Parameters
    ==========
    X_train: pandas df. Original feature data, does not have to be split for supervised learning 
    at this stage. Data must be encoded and normalized before doing dimensionality
    reduction.

    Returns
    ==========
    plot of explained variance
    '''
    pca = PCA().fit(X_train)
    plt.figure()
    cum_sum_pca = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cum_sum_pca)
    plt.xlabel('Number of Components')
    plt.ylabel('Sum of Explaied Variance (%)') # for each component
    plt.title('Explained Variance')
    plt.show()

    
    
def dim_red_pca_only(X_train, num_comps, verbose=True):
    '''
    Reduces dimensionality of original dataset to a predefined number of 
    components. Only uses PCA. The efficacy of the reduction can be assesed via the classification 
    performance with a learner. 

    Parameters
    ==========
    X_train: pandas df. Original feature data, does not have to be split for supervised learning 
    at this stage. Data must be encoded and normalized before doing dimensionality
    reduction.

    num_comps: number of dimensions to reduce original features.

    Returns
    ==========
    X_pca: reduced matrix of size (N, num_comps).

    feats_rank_name: if verbose=True writes csv with the importance of the original
    features in the reduction for the PCA method. We cannot achieve the correspondance
    with the other nonlinear methods but PCA gives a good idea.


    '''
    # pca 
    pca = PCA(n_components=num_comps)
    X_pca = pca.fit_transform(X_train)

    if verbose == True:
        pc_importance = pca.explained_variance_ratio_
        feats_rank = np.argmax(np.abs(pca.components_),axis=1)
        feats_rank_name = pd.DataFrame(X_train.columns[feats_rank].tolist())
        feats_rank_name = pd.concat([feats_rank_name, pd.DataFrame(pc_importance)*100], axis=1)
        feats_rank_name.columns = ['feat name', 'PCA imp weight']
        feats_rank_name.to_csv('pca_feats_rank_name.csv')

    return X_pca
    
    
    
def dim_red_comparison(X_train, y_data, num_comps, verbose=True):
    '''
    Reduces dimensionality of original dataset to a predefined number of 
    components. Different methods are used: PCA, KPCA, Random Projections, and
    LDA. The efficacy of the reduction can be assesed via the classification 
    performance with a learner. 

    Parameters
    ==========
    X_train: pandas df. Original feature data, does not have to be split for supervised learning 
    at this stage. Data must be encoded and normalized before doing dimensionality
    reduction.

    y_data: pandas df. Original label data. Must be encoded. Only used for LDA.

    num_comps: number of dimensions to reduce original features.

    Returns
    ==========
    X_pca, X_kpca, X_rp, X_lda: reduced matrices of size (N, num_comps) for each of the 
    reduction methods.

    feats_rank_name: if verbose=True writes csv with the importance of the original
    features in the reduction for the PCA method. We cannot achieve the correspondance
    with the other nonlinear methods but PCA gives a good idea.


    '''
    # pca 
    pca = PCA(n_components=num_comps)
    X_pca = pca.fit_transform(X_train)

    # kernelized pca
    k_pca = KernelPCA(n_components=num_comps, kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = k_pca.fit_transform(X_train)
    # transform back
#     X_train_kpca_bck = k_pca.inverse_transform(X_kpca) 

    # random projections
    rand_p = random_projection.GaussianRandomProjection(n_components=num_comps)
    X_rp = rand_p.fit_transform(X_train)

    # now do LDA (this is a supervised method for dim red)
    lda = LinearDiscriminantAnalysis(n_components=num_comps)
    X_lda = lda.fit(X_train, y_data).transform(X_train)

    # only pca can give us the importance in the original space because it is 
    # a linear combination 
    if verbose == True:
        pc_importance = pca.explained_variance_ratio_
        feats_rank = np.argmax(np.abs(pca.components_),axis=1)
        feats_rank_name = pd.DataFrame(X_train.columns[feats_rank].tolist())
        feats_rank_name = pd.concat([feats_rank_name, pd.DataFrame(pc_importance)*100], axis=1)
        feats_rank_name.columns = ['feat name', 'PCA imp weight']
        feats_rank_name.to_csv('pca_feats_rank_name.csv')


    return X_pca, X_kpca, X_rp, X_lda
