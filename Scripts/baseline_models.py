import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt

def correlation_info(datamatrix,th,drop,draw):
    print("correlation_info running ... ")
    df_all_data = datamatrix

    corr_matrix = df_all_data.iloc[:,0:(df_all_data.shape[1]-1)].corr()
    if draw:
        sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)

    cormat_melted = []
    for i in range(len(corr_matrix)):
        f1 = corr_matrix.columns[i]
        for j in range(i,len(corr_matrix)):
            f2 = corr_matrix.columns[j]
            cormat_melted.append([f1, f2, corr_matrix.iloc[i,j]])
    cormat_melted = pd.DataFrame(cormat_melted,columns=['f1','f2','values'])
    cormat_melted.head(5)
    cormat_melted_filt = cormat_melted.loc[(cormat_melted['values']>=th) & (cormat_melted['values'] !=1.0)]
    todrop = set(cormat_melted_filt['f2'])

    print ("Correlation filter >" , str(th) , ": " , str(len(todrop)) , " features from the dataset")
    print (todrop)

    if drop ==1:
        return todrop
    else:
        return  []



def grid_search(model , xdata,ydata, cv_, verbose, p_grid=None):

    if model == 'LRLASSO':
        auc_matrix = {}
        for index, regularization_strength in enumerate(p_grid['regularization_strength']):
            model = LogisticRegression(penalty='l1', C=regularization_strength)
            predicted = cross_val_predict(model, xdata,ydata, cv=cv_)
            auc_matrix[regularization_strength] = metrics.accuracy_score(ydata, predicted)
            if verbose == True:
                print('GRID SEARCHING LR: progress: {0:.3f} % ...'.format((index + 1) / (len(p_grid['regularization_strength'])) * 100))
        return auc_matrix
    elif model == 'LSVM':
        auc_matrix = {}
        for index, regularization_strength in enumerate(p_grid['regularization_strength']):
            model = SVC(kernel='linear', C = regularization_strength)
            predicted = cross_val_predict(model,xdata,ydata, cv=cv_)
            auc_matrix[regularization_strength] = metrics.accuracy_score(ydata, predicted)
            if verbose == True:
                print('GRID SEARCHING LSVM: progress: {0:.3f} % ...'.format((index + 1) / (len(p_grid['regularization_strength'])) * 100))
        return auc_matrix
    elif model == 'RF':
            auc_matrix = np.zeros((len(p_grid['n_estimators']), len(p_grid['max_depth']), len(p_grid['min_samples_leaf'])))
            for n_estimator_index, n_estimator in enumerate(p_grid['n_estimators']):
                for max_depth_index , max_depth  in enumerate(p_grid['max_depth']):
                    for min_samples_leaf_index, min_samples_leaf in enumerate(p_grid['min_samples_leaf']):

                        model = RandomForestClassifier(n_jobs=-1, oob_score=True, max_depth=int(max_depth),n_estimators=int(n_estimator),min_samples_leaf=int(min_samples_leaf))
                        predicted = cross_val_predict(model, xdata,ydata, cv=cv_)
                        auc_matrix[max_depth_index, n_estimator_index, min_samples_leaf_index] = metrics.accuracy_score(ydata, predicted)
                        if verbose == True:
                            print('\rGRID SEARCHING RF: progress: {0:.3f} % ...'.format(
                                (max_depth_index * (len(p_grid['max_depth']) * len(p_grid['min_samples_leaf'])) +
                                 n_estimator_index * (len(p_grid['min_samples_leaf'])) +
                                 min_samples_leaf_index
                                 + 1) / (len(p_grid['n_estimators']) * len(p_grid['max_depth']) * len(p_grid['min_samples_leaf'])) * 100))
            return auc_matrix
    elif model == 'XGB':
        auc_matrix = np.zeros((len(p_grid['n_estimators']), len(p_grid['max_depth']), len(p_grid['learning_rate'])))
        for n_estimator_index, n_estimator in enumerate(p_grid['n_estimators']):
            for max_depth_index, max_depth in enumerate(p_grid['max_depth']):
                for learning_rate_index, learning_rate in enumerate(p_grid['learning_rate']):

                    model = XGBClassifier(max_depth=int(max_depth), n_estimators=int(n_estimator), learning_rate=learning_rate)
                    predicted = cross_val_predict(model, xdata,ydata, cv=cv_)
                    auc_matrix[max_depth_index, n_estimator_index, learning_rate_index] = metrics.accuracy_score(ydata, predicted)
                    if verbose == True:
                        print('\rGRID SEARCHING RF: progress: {0:.3f} % ...'.format(
                            ((max_depth_index * (len(p_grid['max_depth']))) +
                             (n_estimator_index * (len(p_grid['n_estimators']))) +
                             (learning_rate_index * (len(p_grid['learning_rate'])))+ 1)
                             / (len(p_grid['n_estimators']) * len(p_grid['max_depth']) * len(
                                p_grid['learning_rate'])) * 100))

        return auc_matrix

