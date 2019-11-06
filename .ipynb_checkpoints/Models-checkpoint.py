# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:49:32 2019

@author: ayrem
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
import pickle
from sklearn.externals import joblib

#from imblearn import over_sampling

from churn import egitimgirdi,egitimcikti,valgirdi,valcikti,egitimveri,validationveri
print(egitimgirdi.shape)
#print(egitimcikti.shape)
"""
ros = over_sampling.RandomOverSampler()
rosegitimgirdi,rosegitimcikti = ros.fit_sample(egitimgirdi, egitimcikti)
print(rosegitimgirdi.shape)

smote = over_sampling.SMOTE()
smoteegitimgirdi,smoteegitimcikti = smote.fit_sample(egitimgirdi, egitimcikti)
print(smoteegitimgirdi.shape)

adasyn = over_sampling.ADASYN()
adasynegitimgirdi,adasynegitimcikti = adasyn.fit_sample(egitimgirdi, egitimcikti)
print(adasynegitimgirdi.shape)
"""
models = []
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DCT",DecisionTreeClassifier()))
models.append(("GNB",GaussianNB()))
models.append(("SVC",SVC()))
models.append(("MLP",MLPClassifier()))
models.append(("ADB",AdaBoostClassifier()))
models.append(('RAF', RandomForestClassifier()))
"""
for name,model in models: -imbalanced sorunu verilerin dengesiz olması sonucun ağır olan kısma göre yoğunluk vermesi
#    egitilmismodel = model.fit(egitimgirdi,egitimcikti)
#    egitilmismodelros = model.fit(rosegitimgirdi,rosegitimcikti)
#    egitilmismodelsmote = model.fit(smoteegitimgirdi,smoteegitimcikti)
    egitilmismodeladasyn = model.fit(adasynegitimgirdi,adasynegitimcikti)
    
    
#    egitimsonuc = egitilmismodel.score(egitimgirdi,egitimcikti)
#    egitimsonucros = egitilmismodelros.score(egitimgirdi,egitimcikti)
#    egitimsonucsmote = egitilmismodelsmote.score(egitimgirdi,egitimcikti)
    egitimsonucadasyn = egitilmismodeladasyn.score(egitimgirdi,egitimcikti)
    
#    valsonuc = egitilmismodel.score(valgirdi,valcikti)
#    valsonucros = egitilmismodelros.score(valgirdi,valcikti)
#    valsonucsmote = egitilmismodelsmote.score(valgirdi,valcikti)
    valsonucadasyn = egitilmismodeladasyn.score(valgirdi,valcikti)
    
    print(name,":")
#    print("ORJ :",egitimsonuc,"VAL - ORJ :",valsonuc)
#    print("ROS :",egitimsonucros,"VAL - ROS :",valsonucros)
#    print("SMOTE :",egitimsonucsmote,"VAL - SMOTE :",valsonucsmote)
    print("ADASYN :",egitimsonucadasyn,"VAL - ADASYN :",valsonucadasyn)
    print("__________________")
"""
"""
for name,model in models:
    egitilmismodel = model.fit(egitimgirdi,egitimcikti)
    egitimsonuc = egitilmismodel.score(egitimgirdi,egitimcikti)
    valsonuc = egitilmismodel.score(valgirdi,valcikti)
    print("Sonuclar: %s: Egitim: %f I Valitations: %f "%(name,egitimsonuc,valsonuc))
"""

#from sklearn.model_selection import GridSearchCV
"""
def knn_param_selection(X, y, nfolds):
    n_neighbors = [2,4,8,16]
    p=[2,3]
    weights=["uniform","distance"]
    metric=["euclidean","manhattan"]
    param_grid = {"n_neighbors": n_neighbors,"p":p,"weights":weights,"metric":metric}
    
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return print(grid_search.best_params_)

knn_param_selection(egitimgirdi,egitimcikti,10)

a=KNeighborsClassifier(n_neighbors=2,p=3,metric="manhattan",weights="uniform").fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
print(a.score(valgirdi,valcikti))
"""
a=KNeighborsClassifier(n_neighbors=2,p=3,metric="manhattan",weights="uniform").fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("KNN confusion_matrix:\n", cm)
print("KNN accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nKNN f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open(filename, 'wb'))
print("\n")

"""
def svc_param_selection(X, y, nfolds):
    Cler = [10,100]
    gammalar=[0.01,0.1]
    kernellar=["linear","rbf"]
    param_grid = {"C": Cler, "gamma": gammalar, "kernel": kernellar}
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return print(grid_search.best_params_)

svc_param_selection(egitimgirdi,egitimcikti,10)

a=SVC(C=100,kernel="rbf",gamma=0.1).fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
print(a.score(valgirdi,valcikti))
"""
a=SVC(C=100,kernel="rbf",gamma=0.1).fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("SVC confusion_matrix:\n", cm)
print("SVC accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nSVC f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open(filename, 'wb'))
print("\n")
"""
def dct_param_selection(X, y, nfolds):
    Max_depth=[1,2,4,8,16,32]
    Min_samples_split=[2,4,8,16,32]
    Min_samples_leaf=[1,2,4,8,16,32]
    criterion=["gini","entropy"]
    param_grid = {"criterion": criterion, "max_depth": Max_depth, "min_samples_split": Min_samples_split,"min_samples_leaf": Min_samples_leaf}
    #param_grid = {"max_depth": Max_depth,"min_samples_split":Min_samples_split,"min_samples_leaf":Min_samples_leaf}

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return print(grid_search.best_params_)

dct_param_selection(egitimgirdi,egitimcikti,10)

a=DecisionTreeClassifier(max_depth=8,min_samples_split=16,min_samples_leaf=32,criterion="gini").fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
print(a.score(valgirdi,valcikti))
"""
a=DecisionTreeClassifier(max_depth=8,min_samples_split=16,min_samples_leaf=32,criterion="gini").fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("Decision Tree confusion_matrix:\n", cm)
print("Decision Tree accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nDecision Tree f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open(filename, 'wb'))
print("\n")
"""
def ram_param_selection(X, y, nfolds):
    n_est = [120, 150]
    Max_depth=[5,15]
    Min_samples_split=[1.0,10]
    #Min_samples_leaf=[1]
    Max_features=["log2","sqrt","auto"]
    param_grid = {"n_estimators": n_est, "max_depth": Max_depth, "min_samples_split": Min_samples_split,"min_samples_leaf": Min_samples_leaf}
    #param_grid = {"n_estimators": n_est, "max_depth": Max_depth,"min_samples_split": Min_samples_split,"min_samples_leaf": Min_samples_leaf}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return print(grid_search.best_params_)

ram_param_selection(egitimgirdi,egitimcikti,10)

a=RandomForestClassifier(n_estimators=120,max_depth=16,min_samples_split=10,max_features="sqrt").fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
print(a.score(valgirdi,valcikti))
"""
a=RandomForestClassifier(n_estimators=250,max_depth=16,min_samples_split=10,max_features="sqrt").fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("Random Forest confusion_matrix:\n", cm)
print("Random Forest accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nRandom Forest f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open("ram_model.pkl", 'wb'))
a=pickle.load(open("ram_model.pkl","rb"))
print("\n")
"""
def adb_param_selection(X, y, nfolds):
    n_est =[120, 100, 300, 500]
    lr = [0.001, 0.01, 0.1, 1]
    param_grid ={"n_estimators": n_est, "learning_rate" : lr}
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
    
print(adb_param_selection(valgirdi,valcikti,3))

a = AdaBoostClassifier(learning_rate=0.1,n_estimators=120).fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
print(a.score(valgirdi,valcikti))

"""  
a=AdaBoostClassifier(learning_rate=0.1,n_estimators=120).fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("AdaBoost confusion_matrix:\n", cm)
print("AdaBoost accuracy_score:", accuracy_score(valcikti, y_pred)),
print("\nAdaBoost f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open(filename, 'wb')) 
print("\n")

a=GaussianNB().fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

cm = confusion_matrix(valcikti, y_pred) 
print("Naive Bayes confusion_matrix:\n", cm)
print("Naive Bayes accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nNaive Bayes f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open(filename, 'wb')) 

"""
Verilerin sonuca etkisi
forest = ExtraTreesClassifier().fit(egitimgirdi,egitimcikti)
print(forest.score(egitimgirdi,egitimcikti))
print(forest.score(valgirdi,valcikti))

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]   
    
print ("Feature Ranking:")
print()
for f in range (egitimgirdi.shape[1]):
    print("%d. feature %d (%f)" % (f+1,indices[f], importances[indices[f]]))
    
plt.figure()
plt.title("Feature importances")
plt.bar(range(egitimgirdi.shape[1]), importances[indices],
        color="r",yerr=std[indices], align ="center")
plt.xticks(range(egitimgirdi.shape[1]), indices)
plt.xlim([-1,egitimgirdi.shape[1]])
plt.show()
"""
"""
#Sınıf ağılıkları
from sklearn.utils import class_weight
print (np.unique(egitimcikti))
agirliklar = class_weight.compute_class_weight("balanced", np.unique(egitimcikti),egitimcikti)
print(agirliklar)
    
from sklearn.ensemble import RandomForestClassifier

a = RandomForestClassifier(random_state=1).fit(egitimgirdi,egitimcikti)
print(a.score(egitimgirdi,egitimcikti))
b = RandomForestClassifier(random_state=1,class_weight={0:0.62,1:2.46}).fit(egitimgirdi,egitimcikti)
print(b.score(egitimgirdi,egitimcikti)) 
"""

    
    