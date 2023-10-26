import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame

data_path = 'C:/Users/laura/OneDrive/Documenti/PhD/ProgettoLorenzo/Data_Code/Data/Carica101_Models/tagging_carica101_'
DATA = pd.read_csv(data_path + 'group_2su3_df_reduced.csv')

# Initialize matrices to store results
r2_mat = np.zeros((5, 5))
rv_mat = np.zeros((5, 5))

# Define matrix indices
d1 = list(range(1, 18))
d2 = list(range(18, 26))
d3 = list(range(26, 32))
d4 = list(range(32, 36))
d5 = list(range(36, 40))

# Combine these indices into a list
matrices = [d1, d2, d3, d4, d5]

#X = np.random.rand(10,10)

# Loop over all possible pairs of matrices
for i in range(len(matrices)):
    for j in range(1, len(matrices)):
        # Extract X and Y for the current pair
        X = DATA.iloc[:, matrices[i]].to_numpy()
        Y = DATA.iloc[:, matrices[j]].to_numpy()
        #ctk = [0, 1, 2, 3, 6, 8, 9, 11, 12, 13, 15, 16]
        #X = X[:, ctk]
        #Y = Y[:, ctk]
        #X = np.random.rand(10,10)
        #Y = X

        # Center X and Y
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)

        # df = pd.read_csv('penguins.csv', delimiter=',')
        # df = df.dropna()
        # X = df[['bill_length_mm','bill_depth_mm']]
        # X_mc = (X-X.mean())/(X.std())
        # Y = df[['flipper_length_mm','body_mass_g']]
        # Y_mc = (Y-Y.mean())/(Y.std())


        # Canonical Correlation Analysis
        cca = CCA(n_components = np.min([X.shape[1], Y.shape[1]]), scale=True)
        cca.fit(X, Y)
        U, V = cca.transform(X, Y)

        A = np.linalg.lstsq(X, U)[0]
        B = np.linalg.lstsq(Y, V)[0]

        # R
        corrs = np.full(U.shape[1], np.nan)
        for c in range(U.shape[1]):
            x = U[:,c]
            y = V[:,c]
            corr = np.corrcoef(x,y, rowvar=False)[0,1]
            corrs[c] = corr

        # Calculate regression coefficients b_coeffs
        b_coeffs = np.linalg.lstsq(U, V)[0]

        # Predict V using U and b_coeffs
        V_predic = U.dot(b_coeffs)

        # Calculate centered predicted Y
        Y_predic = np.dot(V_predic, np.linalg.pinv(B))

        # Calculate SSres
        SSres = np.sum((Y.ravel() - Y_predic.ravel()) ** 2)

        # Calculate SStot
        SStot = np.sum((Y.ravel() - np.mean(Y.ravel())) ** 2)

        # Calculate R-squared
        r2 = 1 - (SSres / SStot)
        r2_mat[i, j] = r2

        #mse = np.zeros(Y.shape[1])
        #for col in range(Y.shape[1]):
            #mse[col] = np.mean(np.square(Y[:, col] - Y_predic[:, col]))

    # Sum the mean squared errors for all columns to get SSres
        #SSres = np.sum(mse)
        #SStot = np.sum((Y - np.mean(Y)) ** 2)
        #r2 = 1 - (SSres / SStot)
        #r2_mat[i, j] = r2

        # Calculate RV coefficient
        AA = X.dot(X.T)
        BB = Y.dot(Y.T)
        AA0 = AA - np.diag(np.diag(AA))
        BB0 = BB - np.diag(np.diag(BB))
        RV = np.trace(AA0.dot(BB0)) / (np.sqrt(np.sum(AA0 ** 2)) * np.sqrt(np.sum(BB0 ** 2)))
        rv_mat[i, j] = RV

print(r2_mat)
print(rv_mat)
print('d')

# r2_mat and rv_mat now contain the R-squared and RV coefficients for all pairs of matrices
