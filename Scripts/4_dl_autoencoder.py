import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold



sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Cancer"]


df= pd.read_csv("~/ghub/Data/df_final_ICUid.csv")
#df= pd.read_csv("test.csv")
df.head(2)
df=df.iloc[:,1:]
df.shape
df.isnull().values.any()



### center data
x_data =df.iloc[:,0:df.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
df.iloc[:,0:df.shape[1]-1] = x_scaled
df.head(10)



count_classes = pd.value_counts(df['IsReadmitted'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


input_dim = df.shape[1]-1
kf = KFold(n_splits=2, random_state=0)
result =[]
df_all_data = df
for train, test in kf.split(df_all_data):
    # create model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    train_data = df_all_data.iloc[train, :]
    test_data = df_all_data.iloc[test, :]
    trainx = train_data.iloc[:, 0:train_data.shape[1]-1]
    trainy = train_data.iloc[:, train_data.shape[1]-1]
    testx = test_data.iloc[:, 0:test_data.shape[1]-1]
    testy = test_data.iloc[:, test_data.shape[1]-1]

    model.fit(trainx, trainy, epochs=1, batch_size=10)
    # evaluate the model
    scores = model.evaluate(testx, testy)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    result.append( scores[1]*100)


X_train, X_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.IsReadmitted == 0]
X_train = X_train.drop(['IsReadmitted'], axis=1)
y_test = X_test['IsReadmitted']
X_test = X_test.drop(['IsReadmitted'], axis=1)

X_train = X_train.values
X_test = X_test.values

X_train.shape


input_dim = X_train.shape[1]
encoding_dim = input_dim


input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu",  activity_regularizer=regularizers.l1(10e-5))(input_layer)
#encoder = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 10), activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)



input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",  activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 100
batch_size = 30


autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1)

plt.plot(history.model.history.history['loss'])
plt.plot(history.model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');



predictions = autoencoder.predict(X_test)
predictions = autoencoder.predict(df.iloc[:,1:df.shape[1]])


df_new_data = pd.DataFrame(predictions)
df_new_data[df_new_data.shape[1]] = df.IsReadmitted

df_new_data.to_csv("test.csv",index=False)


kf = KFold(n_splits=3, random_state=0)
result =[]
df_all_data = df_new_data
for train, test in kf.split(df_new_data):
    # create model
    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    train_data = df_all_data.iloc[train, :]
    test_data = df_all_data.iloc[test, :]
    trainx = train_data.iloc[:, 0:train_data.shape[1]-1]
    trainy = train_data.iloc[:, train_data.shape[1]-1]
    testx = test_data.iloc[:, 0:test_data.shape[1]-1]
    testy = test_data.iloc[:, test_data.shape[1]-1]

    model.fit(trainx, trainy, epochs=1, batch_size=10)
    # evaluate the model
    scores = model.evaluate(testx, testy)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    result.append( scores[1]*100)




#
# mse = np.mean(np.power(X_test - predictions, 2), axis=1)
# error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
#
# error_df.describe()
#
#
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
