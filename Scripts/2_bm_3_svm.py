import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt



def lsvm(xdata,ydata,cv_):
    predicted = cross_val_predict(SVC(), xdata,ydata, cv=cv_)
    print("svm score", metrics.accuracy_score(ydata, predicted))


def grid_search(model , xdata,ydata,cv_, verbose, p_grid=None):
    if model == 'LSVM':
        auc_matrix = np.zeros(len(p_grid['regularization_strength']))
        for index, regularization_strength in enumerate(p_grid['regularization_strength']):
            #model = SVC(kernel='linear', C = regularization_strength)
            model = SVC(kernel='rbf', C=regularization_strength)
            predicted = cross_val_predict(model, xdata,ydata, cv=cv_)
            auc_matrix[index] = metrics.accuracy_score(ydata, predicted)
            if verbose == True:
                print('GRID SEARCHING LSVM: progress: {0:.3f} % ...'.format((index + 1) / (len(p_grid['regularization_strength'])) * 100))
        return auc_matrix


lsvm(X_train,y_train,3)


param_grid = {
    "regularization_strength": [0.001, 0.01, 0.1, 1, 10],
}
scoremat = grid_search('LSVM',X_train,y_train,cv_=3,p_grid=param_grid,verbose=True)
print(scoremat)