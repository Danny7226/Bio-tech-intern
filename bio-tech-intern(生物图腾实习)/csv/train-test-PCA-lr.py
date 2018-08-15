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
from sklearn.metrics import f1_score
import warnings

for i in range(10,20):

    dimension = i
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

    def test_LogisticRegression(X_train, X_test, y_train, y_test):
        # 选择模型
        cls = LogisticRegression()

        # 把数据交给模型训练
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        print("Coefficients:%s, intercept %s"%(cls.coef_,cls.intercept_))
        print("Residual sum of squares: %.2f"% np.mean((cls.predict(X_test) - y_test) ** 2))
        print('Score: %.2f' %  f1_score(y_pred, y_test))
        print("=============Y_test==============")
        print(Y_test.ravel())
        print("============Y_predict============")
        print(cls.predict(X_test))
    if __name__=='__main__':

        test_LogisticRegression(X_train,X_test,Y_train,Y_test) # 调用 test_LinearRegression
