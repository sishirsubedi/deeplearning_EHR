import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.decomposition import PCA


## experimental cancer data
df_all_data = pd.read_csv("~/ghub/df_final_Patientid.csv")
df_all_data = df_all_data.iloc[:,1:]
df_all_data.head(10)
print (df_all_data.shape)

### center data
x_data =df_all_data.iloc[:,0:df_all_data.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
df_all_data.iloc[:,0:df_all_data.shape[1]-1] = x_scaled
df_all_data.head(10)

## visualize data
df_all_data.iloc[:,1:15].diff().hist(color='k', alpha=0.5, bins=50)
df_all_data.iloc[:,1:10].plot.box()

# check correlation
todrop = baseline_models.correlation_info(df_all_data.iloc[:,0:df_all_data.shape[1]],0.8,drop=0,draw=1)
df_all_data.drop(todrop, axis=1, inplace=True)
print (df_all_data.shape)
#Correlation filter > 0.7 :  1  features from the dataset



from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,cross_val_predict
model = LogisticRegression(penalty='l1', C=0.1)
predicted = cross_val_predict(model,df_all_data.iloc[:,0:df_all_data.shape[1]-1],df_all_data.iloc[:,df_all_data.shape[1]],cv=3)
score = metrics.accuracy_score(df_all_data.iloc[:,0:df_all_data.shape[1]-1], predicted)


param_grid = {
    "regularization_strength": [0.01,0.1,1.0,10.0],
}
scoremat = baseline_models.grid_search('LSVM',df_all_data.iloc[:,0:df_all_data.shape[1]],cv_=2,p_grid=param_grid,verbose=True)
print(scoremat)
##array[ 0.82574349  0.82713755  0.83271375  0.83410781]


# # analyze effect of regularization on linear model
# from sklearn.linear_model import LogisticRegression
# LogReg = LogisticRegression()
# LogReg.fit(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1])
# plt.plot(LogReg.coef_[0])
# LogReg2 = LogisticRegression(C=.001, penalty='l1', tol=1e-6)
# LogReg2.fit(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1])
# plt.plot(LogReg2.coef_[0])
# plt.scatter(LogReg.coef_[0],LogReg2.coef_[0])
# np.count_nonzero(LogReg2.coef_[0])
# np.count_nonzero(LogReg.coef_[0])
# #
# #
# import statsmodels.api as sm
# logit = sm.Logit(df_all_data.iloc[:,df_all_data.shape[1]-1],df_all_data.iloc[:,0:df_all_data.shape[1]-1])
# result = logit.fit()
# result.summary()

result={}
result["nbayes"]=0.71
result["logreg"]=0.84
result["svm"]=0.82
result["lasso:.01"]=0.75
result["lasso:.1"]=0.81
result["lasso:1.0"]=0.84
result["RandomFor"]=0.76
result["XGBoost"]=0.80
plt.ylim(0,1)
plt.xlabel('Baseline Models')
plt.ylabel('Accuracy')
plt.bar(result.keys(), result.values(),align='center', alpha=0.5, color='b')

