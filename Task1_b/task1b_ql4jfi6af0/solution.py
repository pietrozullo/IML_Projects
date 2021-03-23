import numpy as np
import pandas as pd 

#Load data

train_path = 'task1b_ql4jfi6af0/train.csv'

training_data = pd.read_csv(train_path)
t = training_data['Id']
y = training_data['y']
X = training_data[['x1','x2','x3','x4','x5']]
y_train = y.to_numpy()
X_train = X.to_numpy()


feature_vector = np.concatenate((X,X**2,np.exp(X),np.cos(X),np.ones([700,1])),axis = 1)

dim = feature_vector.shape[-1]

#SVD decomposition 
R = 0.1 * np.eye(dim)
A = (np.dot(feature_vector.T, feature_vector)) + R
u,s,v = np.linalg.svd(A)
Ainv = np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))

#Compute optimal w
w_hat = np.dot(Ainv, np.dot(feature_vector.T, y_train.T))
y_predict = np.dot(feature_vector, w_hat.T)

