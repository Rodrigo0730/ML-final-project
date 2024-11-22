'''
A file for data exploration
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


train_df = pd.read_csv('train.csv')
train_df.drop(columns=['ID'],inplace=True)
train_df.loc[~train_df.parentspecies.isin(['apin','toluene','decane']),'parentspecies'] = 'unsure'
one_hot = pd.get_dummies(train_df,columns=['parentspecies'])
one_hot.drop(columns=['parentspecies_unsure'],inplace=True)
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train = pd.DataFrame(x_scaler.fit_transform(one_hot.drop(columns=['log_pSat_Pa'])),columns=one_hot.columns.drop('log_pSat_Pa'),index=one_hot.index)
y_train = y_scaler.fit_transform(one_hot.loc[:,'log_pSat_Pa'].to_numpy().reshape(-1,1))

kf = KFold(n_splits=10)
for train_index,test_index in kf.split(X_train):
    rf = RandomForestRegressor()
    rf.fit(X_train.loc[train_index],y_train[train_index])
    print(r2_score(y_scaler.inverse_transform(rf.predict(X_train.loc[test_index]).reshape(-1,1)),y_scaler.inverse_transform(y_train[test_index].reshape(-1,1))))