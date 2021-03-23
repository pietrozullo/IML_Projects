import numpy as np 
import pandas as pd


#Load the file
training_data = pd.read_csv('./train.csv')

#Extract inputs and outputs from training dataframe
X = training_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13']].to_numpy()
y = training_data['y'].to_numpy()


#Import the different lambda values
lam = np.array([0.1,1,10,100,200])

#Import scikitlearn Ridge class 
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=10,random_state=None, shuffle=True)
X_split = kf.get_n_splits(X)
score = np.zeros([5,1])
print('Initial shape',X.shape)
u = 0
for i in lam: 
    RSE = np.zeros([10,1])
    #print(i)
    j = 0
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "\nTEST:", test_index)
        
        #Define the split train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Xfold size', X_train.shape)
        #Define a regression object 
        clf = Ridge(alpha=i)
        #Fit the model
        clf.fit(X_train,y_train)
        #Predict using fitted model
        y_predict = clf.predict(X_test)
        #Compute for each K fold the Root Squared Error  
        RSE[j] = mean_squared_error(y_test,y_predict,squared = False)
        j += 1
    #Average over Kfolds of the RSE    
    MRSE = np.mean(RSE)
    
    #Save the score
    score[u] = MRSE
    u += 1

print('The score is \n',score)
pd.DataFrame(score).to_csv("./Serenissima.csv",header=None,index=None)