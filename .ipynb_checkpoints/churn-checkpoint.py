# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:12:52 2019

@author: ayrem
"""

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df=pd.read_csv("Churn_Modelling.csv",index_col='RowNumber')
df = df.drop(["CustomerId","Surname"],axis=1)
df = df.replace({"Female":0,"Male":1,"France":0,"Germany":1,"Spain":2})


min_max_scaler = MinMaxScaler()

df = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df)

egitimveri,validationveri = train_test_split(df,test_size=0.2,random_state=7)

egitimgirdi = egitimveri.drop(df.columns[10],axis=1)
egitimcikti = egitimveri[10]

valgirdi = validationveri.drop(df.columns[10],axis=1)
valcikti = validationveri[10]

chi2_selector = SelectKBest(chi2, k=5)
X_kbest = chi2_selector.fit_transform(egitimgirdi, egitimcikti)    

print('Original number of features:', egitimgirdi.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

