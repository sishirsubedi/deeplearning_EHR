import pandas as pd
import importlib
from Scripts import baseline_models
importlib.reload(baseline_models)
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df_all_patientdata= pd.read_csv("~/ghub/Data/df_final_ICUid_sample.csv")
df_all_patientdata= pd.read_csv("~/ghub/df_final_Patientid.csv")
df_all_patientdata= pd.read_csv("test.csv")
df_all_patientdata = df_all_patientdata.iloc[:,1:]
df_all_patientdata.head(10)
X_train, X_test, y_train, y_test = train_test_split( df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1], df_all_patientdata.iloc[:,df_all_patientdata.shape[1]-1], test_size=0.33, random_state=42)


param_grid = {
    "n_estimators": [10,20,40],
    "max_depth": [5,10,20],
    "learning_rate": [0.001,0.01,0.1,0.2]}

scoremat = baseline_models.grid_search('XGB',X_train,y_train,cv_=2,p_grid=param_grid,verbose=True)
print(scoremat)


clf = XGBClassifier()
grid_clf = GridSearchCV(clf, param_grid, cv=3)
grid_clf.fit(X_train, y_train)
grid_clf.score(X_test, y_test)
print(grid_clf.best_params_)


model = XGBClassifier(n_estimators=grid_clf.best_params_['n_estimators'],max_depth=grid_clf.best_params_['max_depth'],learning_rate=grid_clf.best_params_['learning_rate'])
model.fit(X_train,y_train)
model.score(X_test,y_test)
predicted = model.predict(X_test)

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


importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
df_all_patientdata.columns[indices]