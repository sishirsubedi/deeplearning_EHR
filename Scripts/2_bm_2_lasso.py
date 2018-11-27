import seaborn as sns
import pandas as pd
import importlib
from Scripts import baseline_models
importlib.reload(baseline_models)
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def logregression(xdata,ydata, cv_,regmethod ,regval):
    model = LogisticRegression(penalty=regmethod,C=regval)
    predicted = cross_val_predict(model, xdata,ydata, cv=cv_)
    print("Logistic regression score", metrics.accuracy_score(ydata, predicted))
    if regval != 1.0:
        return predicted

df_all_patientdata= pd.read_csv("~/ghub/Data/df_final_ICUid_sample.csv")
df_all_patientdata.shape
df_all_patientdata.columns
df_all_patientdata.head(2)

df_all_patientdata.shape
columns_todrop = df_all_patientdata.filter(like='marital_status').columns
df_all_patientdata = df_all_patientdata.drop(columns_todrop,axis =1)
columns_todrop = df_all_patientdata.filter(like='insurance').columns
df_all_patientdata = df_all_patientdata.drop(columns_todrop,axis =1)
columns_todrop = df_all_patientdata.filter(like='los').columns
df_all_patientdata = df_all_patientdata.drop(columns_todrop,axis =1)
columns_todrop = df_all_patientdata.filter(like='gender').columns
df_all_patientdata = df_all_patientdata.drop(columns_todrop,axis =1)
df_all_patientdata.shape
df_all_patientdata.columns
df_all_patientdata.isnull().values.any()
df_all_patientdata = df_all_patientdata.iloc[:,1:]
df_all_patientdata.head(10)


### center data
x_data =df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.StandardScaler().fit(x_data).transform(x_data))
x_scaled.columns = x_data.columns
df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1] = x_scaled
df_all_patientdata.head(10)

X_train, X_test, y_train, y_test = train_test_split( df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1], df_all_patientdata.iloc[:,df_all_patientdata.shape[1]-1], test_size=0.25, random_state=42)

logregression(X_train,y_train,cv_=2,regmethod='l2',regval=1.0)

param_grid = {
    "regularization_strength":[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.04,0.06,0.08, 0.1,0.2,0.4,0.6,0.8,1.0]
}
scoremat = baseline_models.grid_search('LRLASSO',X_train,y_train,cv_=3,p_grid=param_grid,verbose=True)
print(scoremat)

param_grid = {'C': [0.001, 0.01, 0.1, 1] }
clf = LogisticRegression(penalty='l1')
grid_clf = GridSearchCV(clf, param_grid, cv=3)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)
grid_clf.score(X_test,y_test)


penalties = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.04,0.06,0.08, 0.1,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]
for p in penalties:
    model = LogisticRegression(penalty='l1', C=p, fit_intercept=False)
    model.fit(X_train, y_train)
    acc= model.score(X_test, y_test)
    print(p,acc, np.count_nonzero(model.coef_[0]))


model = LogisticRegression(penalty='l1', C=0.006,fit_intercept=False)
model.fit(X_train,y_train)
predicted = pd.DataFrame(model.predict(X_test))
print (np.sum([1 if x == y else 0 for x, y in zip(predicted.values, y_test.values)]) / float(predicted.shape[0]))
model.score(X_test,y_test)
print (np.count_nonzero(model.coef_[0]))
nonzeroindex = [i for i, e in enumerate(model.coef_[0]) if e != 0]
lasso_features = df_all_patientdata.columns[nonzeroindex]
lasso_features


false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, predicted)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

feature_data = df_all_patientdata.loc[:,lasso_features]
control = df_all_patientdata.loc[df_all_patientdata['readmit']==0.0,:]
control = control.loc[:,lasso_features]
case = df_all_patientdata.loc[df_all_patientdata['readmit']==1.0,:]
case = case.loc[:,lasso_features]

fig = plt.figure()
for i in range(0,24):
    ax = plt.subplot(3, 8, i+1 )
    sns.distplot(control.iloc[:,i],label='no-readmit')
    sns.distplot(case.iloc[:,i],label='readmit')
plt.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
