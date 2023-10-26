import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA
from itertools import product
from matplotlib import pyplot as plt
import seaborn as sns
import os

def canoncorrelation(X,Y, center=True):
    
    """
    Performs canonical correlation analysis using sklearn.cross_decomposition CCA package
    
    Inputs:
    - X : matrix of shape n by d1
    - Y : matrix of shape n by d2
    - center: default = True; whether to remove the mean (columnwise) from each column of X and Y

    Outputs:
    - r2 : R-squared (Y*B = V)
    - A : Sample canonical coefficients for X variables
    - B : Sample canonical coefficients for Y variables
    - r : Sample canonical correlations
    - U : canonical scores for X
    - V : canonical scores for Y
    """
    
    # Center X and Y
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

    # Canonical Correlation Analysis
    n_components = np.min([X.shape[1], Y.shape[1]]) # Define n_components as the min rank
    cca = CCA(n_components=n_components, scale=True)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)

    # Get A and B matrices as done by Matlab canoncorr()
    A = np.linalg.lstsq(X, U, rcond=None)[0]
    B = np.linalg.lstsq(Y, V, rcond=None)[0]

    # Calculate R for each canonical variate
    R = np.full(U.shape[1], np.nan)
    for c in range(U.shape[1]):
        x = U[:,c]
        y = V[:,c]
        r = np.corrcoef(x,y, rowvar=False)[0,1]
        R[c] = r

    # Calculate regression coefficients b_coeffs
    b_coeffs = np.linalg.lstsq(U, V, rcond=None)[0]

    # Predict V using U and b_coeffs
    V_pred = np.dot(U, b_coeffs)

    # Calculate centered predicted Y
    Y_pred = np.dot(V_pred, np.linalg.pinv(B))

    # Calculate R-squared
    SSres = np.sum((Y.ravel() - Y_pred.ravel()) ** 2)
    SStot = np.sum((Y.ravel() - np.mean(Y.ravel())) ** 2)
    r2 = 1 - (SSres / SStot)
    
    return r2, A, B, R, U, V

def rv_coeff(X,Y, center=True, zscore=False):
    
    """
    Calculates adjusted RV coefficient between two matrices
    from "Smilde, A. K., Kiers, H. A., Bijlsma, S., Rubingh, C. M., & Van Erk, M. J. (2009). 
    Matrix correlations for high-dimensional data: the modified RV-coefficient. 
    Bioinformatics, 25(3), 401-405."

    Inputs:
    - X : matrix of shape n by d1
    - Y : matrix of shape n by d2
    - center: default = True; whether to remove the mean (columnwise) from each column of X and Y
    - zscore: default = Flase; whether to zscore each column of X and Y

    Outputs:
    - rv : adjusted rv coefficient
    """
    
    # Rescale X and Y 
    if zscore:
        X = stats.zscore(X, axis=0)
        Y = stats.zscore(Y, axis=0)

    # Center X and Y
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

    # Calculate adjusted RV coefficient
    AA = np.dot(X, X.T)
    BB = np.dot(Y, Y.T)
    AA0 = AA - np.diag(np.diag(AA))
    BB0 = BB - np.diag(np.diag(BB))
    rv = np.trace(AA0.dot(BB0)) / (np.sqrt(np.sum(AA0 ** 2)) * np.sqrt(np.sum(BB0 ** 2)))
    
    return rv

def domains_collinearity(domains, path, measure='R2'):
    
    """
    Gets r2 or rv score for all pairs of domains
    
    Inputs:
    - domains : list of domains' names in strings
    - path : path where domains matrices are stored (path includes first part of filename, up to the domain specific)
    - measure : specify whether to get R2 from canonical correlation ('R2') or adjusted RV ('RV'); default = 'R2'
    
    Outputs:
    - r_mat : Results matrix of shape =  n_domains by n_domains with X on the columns and Y on the rows
    """

    # Create domains dictionary
    domains_dict = {d: pd.read_csv(path + '{}.csv'.format(d)).to_numpy() for d in domains}
    
    # Run cca for each pair of domains and save r2     
    domains_combs = list(product(domains, repeat=2)) # get all pairs of domains
    r_mat = np.zeros((len(domains_combs))) # Initialize matrix where to save results

    for c, comb in enumerate(domains_combs):
        X = domains_dict[comb[0]]
        Y = domains_dict[comb[1]]
        
        if measure == 'R2' or measure == 'r2':
            r_mat[c] = canoncorrelation(X, Y)[0]
        elif measure == 'RV' or measure == 'rv':
            r_mat[c] = rv_coeff(X, Y)

    r_mat = r_mat.reshape(5,5).T # Reshape into a n_domain by n_domain matrix
    return r_mat

def plot_canoncorr(R, labels, out_path=False, filename='R2', cbar_lab='R2'):
    
    """
    Plots results with a heatmap

    Inputs:
    - R : matrix of shape n_domains by n_domains
    - labels : array of length n_domains, containing domains' names as strings
    - out_path : path where to save the image; if False, image will be simply shown; default = False
    - filename : figure filename, default = 'R2'
    - cbar_lab : color bar label, indicate measure; default = 'R2'
    
    Outputs:
    - heatmap either saved in outpath or shown
    """
    
    # Create figure
    n_dom = R.shape[0]
    fig = plt.figure(figsize = (n_dom, n_dom))
    
    # Plot heatmap
    sns.heatmap(R, cmap='coolwarm', annot=True, vmin= -1, vmax=1, linewidths=1,  cbar_kws={'label': cbar_lab})
    
    # Set axis and ticks labels
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Predicting')

    # Save or show
    if out_path:
        if not os.path.isdir(out_path): # check if path exists else create directory
            os.mkdir(out_path)
        fig.savefig(out_path + filename, dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()

def sparsity(X, run_cut=[268, 225, 320, 325, 236, 240]):
    
    """
    Measures sparsity of a matrix (domain), for each run

    Inputs:
    - X : domin matrix of shape n_samples by n_columns
    - run_del : list of indices delimiting cut of each run; default=[268, 225, 320, 325, 236, 240]
    
    Outputs:
    - sparsity : np.array of sparsity values of shape = n_runs
    """

    # Initialize Sparsity array
    sparsity = np.full(len(run_cut), np.nan)
    
    # Calculate sparsity for each run
    start = 0
    for r, cut in enumerate(run_cut):
        run = X.iloc[start:start+cut, :]
        sparsity[r] = 1 - np.count_nonzero(run)/(run.shape[0]*run.shape[1])
        start+=cut
    return sparsity

def domains_sparsity(domains, path):
    
    """
    Gets sparsity for each run of each domain
    
    Inputs:
    - domains : list of domains' names in strings
    - path : path where domains matrices are stored (path includes first part of filename, up to the domain specific)
    
    Outputs:
    - spar_mat : Results matrix of shape = n_runs by n_domains
    """
    
    # Initialize results matrix
    spar_mat = np.full((6, len(domains)), np.nan)

    # For each domain, get sparsity of each run
    for d, dom in enumerate(domains):
        domain = pd.read_csv(path + '{}.csv'.format(dom))
        spar_mat[:, d] = sparsity(domain)
    
    spar_mat = pd.DataFrame(spar_mat, columns=domains)
    return spar_mat

def plot_sparsity(R, labels, out_path=False, filename='Sparsity'):
    """
    Plots lineplot from matrix

    Inputs:
    - R : matrix of shape n_domains by n_domains
    - labels : array of length n_domains, containing domains' names as strings
    - out_path : path where to save the image; if False, image will be simply shown; default = False
    - filename : figure filename, default = ''
    
    Outputs:
    - lineplot either saved in outpath or shown
    """

    # Plot all lines
    R.plot.line()

    # Set axis and ticks labels
    plt.xticks(np.arange(0,6), [*range(1,7)], size=10)
    plt.yticks(np.arange(0,1.1,0.2), size=10)
    plt.ylabel('Sparsity', fontweight='bold', size=12)
    plt.xlabel('Run', fontweight='bold', size=12)

    # Remove spines
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set legend
    plt.legend(frameon=False, labelspacing=0.2, loc='upper left', borderaxespad=0.2)

    # Save or show
    if out_path:
        if not os.path.isdir(out_path): # check if path exists else create directory
            os.mkdir(out_path)
        plt.savefig(out_path + filename, dpi=300, facecolor='w', bbox_inches='tight')
    else:
        plt.show()
    return


if __name__ == '__main__':

    # Set data_path as the path where the csv files containing single domain matrices are saved, including first part of filename, up to the domain specification (here I specify 'tagging_carica101_group_2su3_convolved_' for example)
    data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/Domains/tagging_carica101_group_2su3_convolved_'
    
    # Set out_path as the path where to save results figures
    out_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Results/Collinearity/New/'
    
    # Specify domain names in list
    domains_list = ['space_movement', 'agent_objective', 'social_connectivity', 'emotion_expression', 'linguistic_predictiveness']

    sparmat = domains_sparsity(domains_list, data_path)
    plot_sparsity(sparmat, domains_list, out_path)

    R2 = domains_collinearity(domains_list, data_path)
    RV = domains_collinearity(domains_list, data_path, measure='rv')
    
    plot_canoncorr(R2, domains_list, out_path)
    plot_canoncorr(RV, domains_list, out_path, filename='RV', cbar_lab= 'RV')

    

    print('d')