import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np


def naivebayes(df_data,cv):
    print("naivebayes running ... ")
    kf = KFold(n_splits=cv, random_state=0)
    result = []
    for train, test in kf.split(df_data):
        train_data = df_data.iloc[train,:]
        test_data =  df_data.iloc[test,:]

        trainx = train_data.iloc[:,0:df_data.shape[1]-1]
        trainy =   train_data.iloc[:,df_data.shape[1]-1]
        testx = test_data.iloc[:,0:df_data.shape[1]-1]
        testy = test_data.iloc[:,df_data.shape[1]-1]

        clf = GaussianNB()
        clf.fit(trainx, trainy.values)

        yhat = pd.DataFrame(clf.predict(testx), columns=['predict'])
        result.append(np.sum([1 if x == y else 0 for x, y in zip(testy.values, yhat.values)]) / float(len(yhat)))

    print ("Average naivebayes accuracy is:", np.sum(result)/len(result))

naivebayes(df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]],3)