from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pylab as plt


## experimental data
df_all_patientdata= pd.read_csv("~/ghub/Data/df_final_ICUid.csv")
#df_all_data = pd.read_csv("test.csv")
#df_all_patientdata = df_all_patientdata.iloc[:,1:]
df_all_patientdata.head(10)
print (df_all_patientdata.shape)

### center data
x_data =df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
df_all_patientdata.iloc[:,0:df_all_patientdata.shape[1]-1] = x_scaled
df_all_patientdata.head(10)

np.random.seed(0)

kf = KFold(n_splits=2, random_state=0)
result =[]
for train, test in kf.split(df_all_patientdata):
    # create model
    model = Sequential()
    model.add(Dense(356, input_dim=356, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(356, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(178, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(178, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    train_data = df_all_patientdata.iloc[train, :]
    test_data = df_all_patientdata.iloc[test, :]
    trainx = train_data.iloc[:, 0:train_data.shape[1]-1]
    trainy = train_data.iloc[:, train_data.shape[1]-1]
    testx = test_data.iloc[:, 0:test_data.shape[1]-1]
    testy = test_data.iloc[:, test_data.shape[1]-1]

    model.fit(trainx, trainy, epochs=100, batch_size=10)
    # evaluate the model
    scores = model.evaluate(testx, testy)
    #scores = model.evaluate(trainx, trainy)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    result.append( scores[1]*100)
