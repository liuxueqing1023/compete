#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sun Sep 16 17:32:29 2018

@author: 刘雪晴
"""

#1.导入数据集
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
    
trainX=pd.read_csv('trainx.csv')
trainY=pd.read_csv('trainy.csv')
testX=pd.read_csv('testx.csv')

#2.对trainX和testX进行缺失值处理
from sklearn import preprocessing
pre=preprocessing.Imputer(missing_values='NaN')
#fit_transform先拟合数据再标准化
trainX=pre.fit_transform(trainX)
testX=pre.fit_transform(testX)


#3.k折交叉验证,划分训练集和测试集
X =np.array(trainX)
Y =np.array(trainY)

kf = KFold(X.shape[0], n_folds=20)
KFold(X.shape[0],n_folds=20)
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
#用随机森林
from sklearn.ensemble import RandomForestClassifier
alg = RandomForestClassifier(random_state=1,n_estimators=100)
alg.fit(X_train, Y_train)
alg.predict
print('准确率：',alg.score(X_test, Y_test))
