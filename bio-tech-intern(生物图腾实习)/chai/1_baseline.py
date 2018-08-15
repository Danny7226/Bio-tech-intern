# -*- coding: utf-8 -*-
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn import preprocessing
#from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy import stats


data_file = open("预测医院获得性AKI_神经网络.csv")
data = pd.read_csv(data_file)

def pre_process(data):
    """
    :param data: 输入的数据集
    :return:
    """
    #把数据中空的填充为nan
    for column in data.columns:
        data[column] = data[column].apply(lambda x: np.nan if x ==" "else x)

    #删掉缺失率太多的列
    drop_feat = ["AKI前造影剂","ProNTBNPmax_preAKI","氯max_preAKI","糖化血红蛋白max_preAKI","D二聚体max_preAKI"]
    data.drop(drop_feat,axis=1,inplace=True)
    #糖尿病中有"#VALUE!"值，填0
    data["糖尿病"] = data["糖尿病"].apply(lambda x : 0 if str(x)=="#VALUE!" else x)
    #先删掉有缺失的行,简单粗暴，后面再改
    data.dropna(inplace=True,axis=0)
    #入院科室先labelencoder
    # le = preprocessing.LabelEncoder()
    # data[["入院科室"]] = le.fit_transform(data[["入院科室"]])
    # 这些列其中有些是str，有些是int64，懒得一个个看，直接全部转int64
    le_columns = ['贫血分层', '性别', '年龄分层', '年龄', '高血压',
           '人工确认CH1_心肌梗塞', '人工确认CH2_心衰', '人工确认CH3_PVD', '人工确认CH4_CVD',
           '人工确认CH5_Dementia', '人工确认CH6_CPD', '人工确认CH7_CTD', '人工确认CH8_Ulcer',
           '人工确认CH9_MLD', '人工确认CH10_DM', '人工确认CH13_DMamp并发症', '糖尿病', '人工确认CH11_偏瘫',
           '人工确认CH12_中重度肾病', '人工确认CH14_Tumor', '人工确认CH15_Leukemia',
           '人工确认CH16_Lymphoma', '人工确认CH17_SLD', '人工确认CH18_转移瘤', '人工确认CH19_AIDS',
           '氨基糖甙_preAKI', '糖肽_preAKI', '抗真菌_preAKI', '抗病毒_preAKI', 'β内酰胺_preAKI',
           '其它抗生素_preAKI', '利尿剂或脱水_preAKI', 'NSAID_preAKI', 'AKI前造影剂', '中药_preAKI',
           'RASI_preAKI', '止血药_preAKI', '补液或改善循环_preAKI', '中药针剂_preAKI',
           '抗心衰_preAKI', 'PPI_preAKI','CKD分期']
    for column in le_columns :
        if column not in drop_feat:
            data[[column]]=data[[column]].astype("int64")
    #合并患者医院id
    id = ['患者编号','医院编号']
    for column in id :
        if column not in drop_feat:
            data[[column]]=data[[column]].astype("str")
    data["患者_医院"] = data["患者编号"] + '_' + data["医院编号"]
    data.drop(["患者编号", "医院编号"], axis=1, inplace=True)
    #入院科室分列
    data["1"]=1   #这个操作，看看就好，还没想到优化的方法，就是让病人入院科室填1
    data["入院科室"] = data["入院科室"].apply(lambda x : "科室："+x)
    data1 = data.pivot(index="患者_医院", values="1", columns="入院科室").reset_index()
    data = pd.merge(data,data1,on="患者_医院")
    data.fillna(0,inplace=True)
    data.drop(["入院科室","1"],axis=1,inplace=True)
    #数值型好像就剩下这三列了，考虑box-cox变换？
    float_cols = ['钾max_preAKI', '钠max_preAKI', '尿酸max_preAKI']
    data[float_cols] = data[float_cols].astype(float)
    #删离群值
    data = data[data['钾max_preAKI']<30]
    # for i in float_cols:
    #     data[i+"_log"] = data[i].apply(lambda x: np.log(x))
    #归一化这三列
    scaler= sklearn.preprocessing.MinMaxScaler()
    #scaler = sklearn.preprocessing.StandardScaler()
    data[float_cols] = scaler.fit_transform(data[float_cols])
    return data


data= pre_process(data)

#在这里去掉得分最低的那几个特征
drop_feat = ['人工确认CH5_Dementia','人工确认CH11_偏瘫','人工确认CH15_Leukemia','人工确认CH16_Lymphoma',
              '人工确认CH17_SLD','人工确认CH19_AIDS','糖肽_preAKI','抗真菌_preAKI','抗病毒_preAKI','中药_preAKI',
             '抗心衰_preAKI','人工确认CH8_Ulcer','人工确认CH9_MLD','科室：删除','科室：皮肤科','科室：综合','人工确认CH1_心肌梗塞',
             '人工确认CH7_CTD','人工确认CH13_DMamp并发症','人工确认CH18_转移瘤','科室：其它']
data.drop(drop_feat,axis=1,inplace=True)

feature = [i for i in data.columns if i not in ["患者_医院","AKI"]]
x = data[feature]
y = data["AKI"]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

start=time.time()
lgb=LGBMClassifier(random_state=0,learning_rate=0.05,num_leaves=40,max_depth=10,n_estimators=50,min_child_weight=8,subsample=0.6, colsample_bytree=0.7,reg_alpha=0, reg_lambda=0.5)
lgb.fit (X_train,y_train)
end=time.time()
y_pred_lgb=lgb.predict_proba(X_test)[:,1]
print("lgb:",metrics.roc_auc_score(y_test,y_pred_lgb))
print("time",end-start)
lgb_feature_impotance=[(i[0],i[1]) for i in zip(X_train.columns,lgb.feature_importances_)]
lgb_feature_impotance.sort(key=lambda x:x[1])

# for i in lgb_feature_impotance:
#     print(i)

