import pandas as pd
import seaborn as sns
import importlib
from Scripts import baseline_models
importlib.reload(baseline_models)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import random

### get all data and sample 50:50 for modeling
df_all= pd.read_csv("~/ghub/Data/df_final_ICUid.csv")
df_all.shape
df_all.columns
df_all.isnull().values.any()
#df_all = df_all.iloc[:,1:]
df_all.head(1)

# df_icustays= pd.read_csv("~/ghub/Data/ICUSTAYS.csv")
# df_icustays.head(2)
# df_icustays = df_icustays.iloc[:,[1,3]]
#


df_all_one = df_all[df_all.IsReadmitted ==1]
df_all_one.shape
df_all_zero = df_all[df_all.IsReadmitted ==0]
df_all_zero = df_all_zero.iloc[random.sample([x for x in range(46473)],6121),:]
df_all_zero.shape

df_all_patientdata = pd.concat([df_all_one,df_all_zero],axis=0)
df_all_patientdata.shape
df_all_patientdata.head(1)
df_all_patientdata = df_all_patientdata.sample(frac=1).reset_index(drop=True)
df_all_patientdata.to_csv("~/ghub/Data/df_final_ICUid_sample.csv",index=False)


############################################ finish sampling ######


# df_all_patientdata= pd.read_csv("~/ghub/Data/df_final_ICUid_sample.csv")
# df_all_patientdata.head(2)
# df_all_patientdata.shape
# df_all_patientdata.columns
#
#
# ### center data
# x_data =df_all_patientdata.iloc[:,1:df_all_patientdata.shape[1]-1]
# x_scaled = pd.DataFrame(preprocessing.StandardScaler().fit(x_data).transform(x_data))
# x_scaled.columns = x_data.columns
# df_all_patientdata.iloc[:,1:df_all_patientdata.shape[1]-1] = x_scaled
# df_all_patientdata.head(10)
#
# X_train, X_test, y_train, y_test = train_test_split( df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1], df_all_patientdata.iloc[:,df_all_patientdata.shape[1]-1], test_size=0.3, random_state=42)
#


df_train= pd.read_csv("~/ghub/Data/train_data.csv")
df_train= pd.read_csv("~/ghub/Data/ae_train_data.csv")
df_train.head(2)
df_train.shape
X_train = df_train.iloc[:,1:df_train.shape[1]-1]
y_train = df_train.iloc[:,df_train.shape[1]-1]


df_test= pd.read_csv("~/ghub/Data/test_data.csv")
df_test.head(2)
df_test.shape
X_test = df_test.iloc[:,1:df_test.shape[1]-1]
y_test = df_test.iloc[:,df_test.shape[1]-1]



model = LogisticRegression(penalty='l1', C=0.01,fit_intercept=False)
model.fit(X_train,y_train)
model.score(X_test,y_test)
print (np.count_nonzero(model.coef_[0]))
nonzeroindex = [i for i, e in enumerate(model.coef_[0]) if e != 0]
lasso_features = X_train.columns[nonzeroindex]
lasso_coef = model.coef_[0][nonzeroindex]
df_lasso = pd.DataFrame(list(zip(lasso_features,lasso_coef)))
df_lasso.columns =['lasso','value']
df_lasso.value = abs(df_lasso.value)
df_lasso.sort_values('value',inplace=True,ascending=False)
df_lasso = pd.DataFrame(df_lasso['lasso'])

#rf
best_n_estimators=40
best_max_depth=10
best_min_samples_leaf=25
model = RandomForestClassifier(n_estimators=best_n_estimators,max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf)
model.fit(X_train,y_train)
model.score(X_test,y_test)
importances = model.feature_importances_
indices = np.argsort(importances)[-29:]
df_rf = pd.DataFrame(X_train.columns[indices])
df_rf.columns =['rf']

#xgb
best_params_n_estimators=25
best_max_depth=10
best_learning_rate = 0.2
model = XGBClassifier()
model.fit(X_train,y_train)
model.score(X_test,y_test)
importances = model.feature_importances_
indices = np.argsort(importances)[-29:]
df_xgb = pd.DataFrame(X_train.columns[indices])
df_xgb.columns = ['xgb']

#svm
k=1
model = LinearSVC()
rfe = RFE(model, k)
rfe = rfe.fit(X_train,y_train )
lsvc_ranking =[]
for x,d in zip(rfe.ranking_,X_train.columns):
    lsvc_ranking.append([d,x])
lsvc_ranking = pd.DataFrame(lsvc_ranking,columns=['features','lsvc'])
lsvc_ranking.sort_values('lsvc',inplace=True)
df_svm = lsvc_ranking.features[0:29].values
df_svm=pd.DataFrame(df_svm)
df_svm.columns= ['svm']

#### combine all features
combine = pd.concat([df_lasso.lasso, df_xgb.xgb, df_rf.rf,df_svm.svm],axis=1)
combine.head(1)
combine.to_csv("~/ghub/Data/combine_features.csv",index = False)


combine = pd.read_csv("~/ghub/Data/combine_features.csv")
combine.head(2)


common_indices = combine.lasso.isin(combine.rf) & combine.lasso.isin(combine.xgb) #&combine.lasso.isin(combine.svm)
common_features = list(combine.lasso[common_indices])
len(common_features)
common_features

all_features = np.unique(combine.values.flatten())
len(all_features)

uncommon_features = []
for x in all_features:
    if x not in common_features:
        uncommon_features.append(x)
len(uncommon_features)

#
# df_all_patientdata_small = df_all_patientdata.loc[:,all_features]
# df_all_patientdata_small['IsReadmitted'] = df_all_patientdata.IsReadmitted
# df_all_patientdata_small.head(2)

# X_train, X_test, y_train, y_test = train_test_split( df_all_patientdata_small.iloc[:,0:df_all_patientdata_small.shape[1]-1], df_all_patientdata_small.iloc[:,df_all_patientdata_small.shape[1]-1], test_size=0.25, random_state=42)
#

df_nlp = pd.read_csv("~/ghub/Data/nlp_output.csv")
df_nlp.head(2)
nlp_features = list(common_features)
for items in df_nlp.columns[0:10]: nlp_features.append(items)
len(nlp_features)


df_train= pd.read_csv("~/ghub/Data/train_data.csv")
#df_train= pd.read_csv("~/ghub/Data/ae_train_data.csv")
df_train.head(2)
df_train.shape
X_train = df_train.iloc[:,0:df_train.shape[1]-1]
y_train = df_train.iloc[:,df_train.shape[1]-1]

X_train_nlp = pd.merge(X_train,df_nlp, on='icustay_id',how='left')
X_train_nlp.columns
X_train_nlp.shape
X_train_nlp = X_train_nlp.iloc[:,1:X_train_nlp.shape[1]-1]
X_train_nlp.head(1)
X_train = X_train_nlp.loc[:,nlp_features]
X_train.columns
X_train.shape


df_test= pd.read_csv("~/ghub/Data/test_data.csv")
df_test.head(2)
df_test.shape
X_test = df_test.iloc[:,0:df_test.shape[1]-1]
y_test = df_test.iloc[:,df_test.shape[1]-1]


X_test_nlp = pd.merge(X_test,df_nlp, on='icustay_id',how='left')
X_test_nlp.columns
X_test_nlp = X_test_nlp.iloc[:,1:X_test_nlp.shape[1]-1]
X_test_nlp.head(1)
X_test = X_test_nlp.loc[:,nlp_features]

X_test.fillna(0,inplace=True)
X_train.fillna(0,inplace=True)


current = 0
i=0
j=0
while j<25:

    #logreg = XGBClassifier()
    logreg = LogisticRegression()
    #logreg = RandomForestClassifier()
     #LinearSVC()

    logreg.fit(X_train, y_train)

    current = logreg.score(X_test, y_test)


    adding_features = uncommon_features[i]
    adding_matrix = df_train.loc[:,adding_features]
    X_train_new = pd.concat([X_train,adding_matrix],axis=1)

    adding_matrix2 = df_test.loc[:, adding_features]
    X_test_new = pd.concat([X_test, adding_matrix2], axis=1)
    logreg.fit(X_train_new,y_train)

    i+= 1
    if current< logreg.score(X_test_new,y_test):


        print(adding_features)
        print(len(all_features[0:i]), logreg.score(X_test_new, y_test))
        X_train = X_train_new
        X_test = X_test_new

        findex = uncommon_features.index(adding_features)

        del uncommon_features[findex]
        i=0

        j+= 1


X_train.columns
final_features = pd.DataFrame(X_train.columns)
final_features.to_csv("final_features.csv",index=False)


df_all_patientdata_small = df_all_patientdata.loc[:,df_rf.values.flatten()]
df_all_patientdata_small['IsReadmitted'] = df_all_patientdata.IsReadmitted
df_all_patientdata_small.head(2)


logreg = RandomForestClassifier()
logreg.fit(X_train.loc[:,df_rf.values.flatten()],y_train)
print(len(df_rf.values),logreg.score(X_test.loc[:,df_rf.values.flatten()],y_test))

logreg = RandomForestClassifier()
logreg.fit(X_train,y_train)
print(len(df_rf.values),logreg.score(X_test,y_test))

df_1= pd.read_csv("~/ghub/Data/D_ITEMS.csv")
df_1.head(2)
df_1.columns
df_1 =df_1.iloc[:,[1,2]]

df_2= pd.read_csv("~/ghub/Data/D_LABITEMS.csv")
df_2.head(2)
df_2.columns
df_2 =df_2.iloc[:,[1,2]]

df_label = pd.concat([df_1,df_2])
df_label.head(2)
df_label.shape




current = 0
i=0
while i<50:
    #logreg = XGBClassifier(n_estimators=best_params_n_estimators,max_depth=best_max_depth,learning_rate=best_learning_rate)
    logreg = LogisticRegression()
    #logreg = DecisionTreeClassifier()

    logreg.fit(X_train, y_train)
    current = logreg.score(X_test, y_test)


    adding_features = uncommon_features[i]
    adding_matrix = df_train.loc[:,adding_features]
    X_train_new = pd.concat([X_train,adding_matrix],axis=1)
    adding_matrix2 = df_test.loc[:, adding_features]
    X_test_new = pd.concat([X_test, adding_matrix2], axis=1)
    logreg.fit(X_train_new,y_train)

    i+= 1
    if current< logreg.score(X_test_new,y_test):
        print(adding_features)
        print(len(all_features[0:i]), logreg.score(X_test_new, y_test))
        X_train = X_train_new
        X_test = X_test_new
        i=0


X_train.to_csv("xtrain_20features.csv",index=False)
X_test.to_csv("xtest_20features.csv",index=False)

y_train.to_csv("ytrain_20features.csv",index=False)
y_test.to_csv("ytest_20features.csv",index=False)