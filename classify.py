"""
classify conditions based on correlation
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from itertools import combinations


def load_corr(atlas, corr_type='correlation', sess_types=['pain','rest']):
    """load correlation and create labels"""
    static_dir = os.path.join('..', 'output', atlas, 'static_corr')
    sess_ls = []
    label_ls = []
    for n, sess in enumerate(sess_types):
        sess_file = corr_type+'_'+sess+'.npy'
        tmp = np.load(os.path.join(static_dir, sess_file))
        sess_ls.append(tmp)
        label_ls.append(n*np.ones(tmp.shape[0]))
    sess_mat = np.concatenate(sess_ls)
    label_mat = np.concatenate(label_ls)
    return sess_mat, label_mat

def reg_cv(X, y, svc=False):
    """regression with leave one out cv"""
    loo = LeaveOneOut()
    scores = []
    for train, test in loo.split(X):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        if svc:
            # svc
            clf = LinearSVC().fit(X_train, y_train)
        else: 
            # regression
            clf = linear_model.LogisticRegression()
            clf.fit(X_train, y_train)
        # make prediction
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    mean_score = np.mean(scores)
    print(f'mean accuracy={mean_score:.3f}')
    return mean_score



def dcc_reshape(dcc_dir, time_bins=10):
    """reshape dcc mat to 2d into time bins"""
    for f in os.listdir(dcc_dir):
        if f.endswith('.npy'):
            print(f)


def get_sample_cov_matrix(X):
    """
    Returns the sample covariance matrix of data X
    Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable
    Returns:
    (numpy array of floats)   : Covariance matrix
    """

    # Subtract the mean of X
    X = X - np.mean(X, 0)
    # Calculate the covariance matrix (hint: use np.matmul)
    cov_matrix =  1 / X.shape[0] * np.matmul(X.T, X)

    return cov_matrix

def sort_evals_descending(evals, evectors, n_components=None):
    """
    Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
    eigenvectors to be in first two quadrants (if 2D).

    Args:
    evals (numpy array of floats)    : Vector of eigenvalues
    evectors (numpy array of floats) : Corresponding matrix of eigenvectors
                                        each column corresponds to a different
                                        eigenvalue

    Returns:
    (numpy array of floats)          : Vector of eigenvalues after sorting
    (numpy array of floats)          : Matrix of eigenvectors after sorting
    """

    index = np.flip(np.argsort(evals))
    if n_components is not None:
        index = index[:n_components]
    evals = evals[index]
    evectors = evectors[:, index]
    return evals, evectors

def change_of_basis(X, W):
    """
    Projects data onto a new basis.

    Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

    Returns:
    (numpy array of floats)   : Data matrix expressed in new basis
    """

    Y = np.matmul(X, W)
    return Y


def pca(X, n_components):
    """
    Performs PCA on multivariate data.
    Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable
    Returns:
    (numpy array of floats)   : Data projected onto the new basis
    (numpy array of floats)   : Vector of eigenvalues
    (numpy array of floats)   : Corresponding matrix of eigenvectors
    """

    # Subtract the mean of X
    X = X - np.mean(X, axis=0)
    # Calculate the sample covariance matrix
    cov_matrix = get_sample_cov_matrix(X)
    # Calculate the eigenvalues and eigenvectors
    evals, evectors = np.linalg.eigh(cov_matrix)
    # Sort the eigenvalues in descending order
    evals, evectors = sort_evals_descending(evals, evectors, n_components)
    # Project the data onto the new eigenvector basis
    score = change_of_basis(X, evectors)

    return score, evectors, evals

if __name__=="__main__":
    result = []
    combs = list(combinations(['pain','relief','rest'],2))
    for corr in ['correlation', 'partial', 'precision']:
        for comb in combs:
            for n in range(1,30):
                X,y = load_corr('yeo', corr_type=corr, sess_types=comb)
                X_pca, _, _ = pca(X, n_components=n)
                accu = reg_cv(X_pca, y, svc=False)
                result.append(pd.DataFrame({'combination':[str(comb)], 'correlation': corr, 'pca_num': n, 'accuracy':accu}))
    df = pd.concat(result)
    df.to_csv('./result/static_corr_cv.csv')