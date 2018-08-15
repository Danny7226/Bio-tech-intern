import glob
import radiomics
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd


dt = pd.read_csv('train.csv',encoding='gbk')

label = dt.loc[:,['MRN','moveornot']]

data = pd.read_csv('train_data_T2.csv')

df = pd.merge(label,data, how = 'inner', on = 'MRN')

df.to_csv('train_data_T2.csv',index = None)

# df1 = pd.DataFrame({'id':[1,2,3],'F1':[0.7,1.7,3.2],'F2':[1.3,2.4,3.7]},
# columns = ['id','F1','F2'])
#
# df2 = pd.DataFrame({'id':[1,3,4],'F3':[1.1,2.2,3.3]},
# columns = ['id','F3'])
#
# print(df1)
# print(df2)
#
# df = pd.merge(df1,df2,how = 'inner', on = 'id')
# print(df)
