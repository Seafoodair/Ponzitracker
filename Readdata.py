#encoding=utf-8
"""
@author=gang wang

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
#matplotlib inline

#method used for preprocessing
from sklearn.preprocessing import StandardScaler

#models used for training/fitting data
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
# import xgboost

#methods for training and optimizing model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,auc,roc_curve,recall_score,classification_report

#methods for resampling
# from imblearn.over_sampling import SMOTE, RandomOverSampler
# from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler
# from imblearn.combine import SMOTEENN,SMOTETomek
#from imblearn.ensemble import BalanceCascade  not found

#ensemble methods
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'  # 字体 你也可以设置成 新罗马 plt.rcParams['font.family'] = 'Times New Roman'
# print("hello")
df_ponzi = pd.read_csv('./leancy_txhistory.csv')
df_ponzi['date'] = pd.to_datetime(df_ponzi['date'].astype(str), format='%Y%m%d')
# print(df_ponzi.head(10)) #print  before 10 line information
# plt.figure(figsize=(10,6))
#date_range = (df_ponzi['date'] > '2013-1-1') & (df_ponzi['date'] <= '2014-4-1')
#
# df_daterange = df_ponzi[date_range]
# #plt.title("Balance over time",fontsize=20)
# #plt.xlabel("Date",fontsize=20)
# plt.ylabel("BTC",fontsize=20)
# plt.plot(df_daterange['date'], df_daterange['bal'],color='#DA70D6')
# # 设置坐标轴字体大小
# plt.tick_params(axis='x', labelsize=20,rotation=45)  # 设置x轴的字体大小
#
# plt.tick_params(axis='y', labelsize=20)  # 设置y轴的字体大小
# plt.figure(figsize=(10,6))
# counts = df_daterange['date'].value_counts(sort=False)
# #plt.title("Transactions per month",fontsize=20)
# plt.xlabel("Month")
# plt.ylabel("Transaction Counts",fontsize=20)
# plt.tick_params(axis='x', labelsize=20,rotation=45)  # 设置x轴的字体大小
# plt.tick_params(axis='y', labelsize=20)  # 设置y轴的字体大小
# plt.bar(counts.index,counts,color='aqua')
# plt.show()
# #
#
plt.figure(figsize=(10,6))
date_range = (df_ponzi['date'] > '2014-1-1') & (df_ponzi['date'] <= '2014-4-1')
df_inout = df_ponzi[date_range]

#plt.title("BTC In vs. Out")
#plt.xlabel("Date")
plt.ylabel("BTC",fontsize=20)
plt.tick_params(axis='x', labelsize=20,rotation=45)  # 设置x轴的字体大小
plt.tick_params(axis='y', labelsize=20)  # 设置y轴的字体大小
plt.plot(df_inout['date'], df_inout['btc_out'],'r',df_inout['date'], df_inout['btc_in'],'g')

# df = pd.read_csv('./final_aggregated_dataset.csv')
# df.head(10)
# df = df.drop('address', 1)
# X = df.iloc[:, df.columns != 'class']
# y = df.iloc[:, df.columns == 'class']
# ax = sns.distplot(df['lifetime'])
# ax.set_title('Wallet Lifetime Distribution')
# ax.set_xlabel('Lifetime')
# ax.set_ylabel('Frequency')
plt.show()