# -*-coding: utf-8 -*-
import glob
import radiomics
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV,LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
import matplotlib as mpl
import warnings


dimension = 5
pca=PCA(n_components = dimension)


##################################train-data-process
train_raw_data = pd.read_csv('train/train_data_T1.csv')
before1 = train_raw_data.iloc[:,2:]    #PCA之前数据（去除了MRN与label）
# print(before1)
arrs1 = before1.values
train_data = pca.fit_transform(arrs1)
train_label = train_raw_data.iloc[:,1]
train_label = train_label.values

#n = len(data)
#print(n)
#arrs = arrs.reshape(n,-1)
# print(arrs1)
# print('------')

################################test-data-process
test_raw_data = pd.read_csv('test/test_data_T1.csv')
before2 = test_raw_data.iloc[:,2:] #PCA之前测试数据（去掉MRN与label）
arrs2 = before2.values
test_data = pca.fit_transform(arrs2)
test_label = test_raw_data.iloc[:,1]
test_label = test_label.values


################################
X_train = train_data
X_test = test_data
Y_train = train_label
Y_test = test_label

#对数据的训练集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)     #先拟合数据在进行标准化

lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.logspace(-2,2,20),cv=2,penalty="l2",solver="lbfgs",tol=0.01)
re = lr.fit(X_train,Y_train)

r = re.score(X_train,Y_train)
print('R(score):',r)
print('coefficient:',re.coef_)
print("intercept:",re.intercept_)
print("稀疏化特征比率:%.2f%%" %(np.mean(lr.coef_.ravel()==0)*100))
print("=========sigmoid函数转化的值，即：概率p=========")
print(re.predict_proba(X_test))     #sigmoid函数转化的值，即：概率p

#模型的保存与持久化
from sklearn.externals import joblib
joblib.dump(ss,"logistic_ss.model")     #将标准化模型保存
joblib.dump(lr,"logistic_lr.model")     #将训练后的线性模型保存
joblib.load("logistic_ss.model")        #加载模型,会保存该model文件
joblib.load("logistic_lr.model")

#预测
X_test = ss.transform(X_test)       #数据标准化
Y_predict = lr.predict(X_test)      #预测

print("=============Y_test==============")
print(Y_test.ravel())
print("============Y_predict============")
print(Y_predict)
