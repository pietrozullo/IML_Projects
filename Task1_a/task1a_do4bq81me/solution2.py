import numpy as np
import pandas as pd
from scipy import optimize

from sklearn.model_selection import KFold

#load data from csv file
data = pd.read_csv("train.csv") 

# divide features and labels
X = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']].to_numpy()
y = data[['y']].to_numpy()

# split data into 10 subsets and generate fold indices
kf = KFold(n_splits=10)
X_split = kf.get_n_splits(X)

# initialize score vector
score = np.zeros([5,1])

# vector of lambdas
lam = np.array([0.1,1,10,100,200])

# Sweep through values of lambda
r = 0
for i in lam: 
    RSE = np.zeros([10,1])
    j = 0
    for train_index, test_index in kf.split(X):
        
        #Define the split train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

		# Train the Ridge estimator
        dim = X_train.shape[-1]
		#SVD decomposition 
        A = (np.dot(X_train.T, X_train) + i * np.eye(dim))
        u,s,v=np.linalg.svd(A)
        Ainv=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
        #Compute optimal w
        w_hat = np.dot(Ainv, np.dot(X_train.T, y_train))
        y_predict = np.dot(X_test, w_hat)
        print('Correct w_hat shape is', w_hat.shape)
        point_error = y_predict-y_test
        SE = np.dot(point_error.T,point_error)
        MSE = SE / 10

        RSE[j] = np.sqrt(MSE)

        j += 1
    #Average over Kfolds of the RSE    
    MRSE = np.mean(RSE)
    #print(MRSE)
    #Save the score
    score[r] = MRSE
    r += 1

print(score)
# export to csv
pd.DataFrame(score).to_csv("./Serenissima2.csv",header=None,index=None)









