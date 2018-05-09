from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model, load_model
from keras import optimizers
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from Scripts import baseline_models
from sklearn.model_selection import cross_val_score

RANDOM_SEED = 42

df = pd.read_csv("Data/train_data.csv")
# df = pd.read_csv("Data/test_data.csv")
data = df.iloc[:, 1:]
baseline_models.center(data)

# use unlabeled data to train AE
df2 = data.drop(['IsReadmitted'], axis=1)
X_train, X_test = train_test_split(df2, test_size=0.05, random_state=RANDOM_SEED)
X_train = np.array(X_train)
X_test = np.array(X_test)

# X_train = X_train[X_train.IsReadmitted == 0]

# # add random noise to get more robustness
# noise_factor = 0.5
# x_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
# x_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
# X_train = x_train_noisy.values
# X_test = x_test_noisy.values
# # not working good, seems the network cannot reverse the random noise


# autoencoder architecture
middle = 500
hidden = 340
input_dim = X_train.shape[1]  # 340
input_layer = Input(shape=(input_dim,))
encoded = Dense(hidden, activation="relu", kernel_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(middle, activation="relu")(encoded)

decoded = Dense(hidden, activation='relu')(encoded)
decoded = Dense(input_dim, activation='relu')(decoded)  # cannot use tanh here

encoder = Model(input_layer, encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)

lr = 0.001
nb_epoch = 300
batch_size = 64

# start train
adam = optimizers.Adam(lr=lr)
autoencoder.compile(optimizer=adam,
                    loss='mean_squared_error',
                    metrics=['accuracy'])

log_dir = '/tmp/tensorflow_logs/ae2'
history = autoencoder.fit(X_train, X_train,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=2,
                          callbacks=[TensorBoard(log_dir=log_dir)])

autoencoder.save('autoencoder_7853.hdf5')

autoencoder = load_model('autoencoder_7853.hdf5')

from keras.utils import plot_model
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)

# encoded and test
predictions = encoder.predict(df2)
df_new_data = pd.DataFrame(predictions)
df_new_data[df_new_data.shape[1]] = df.IsReadmitted
id_ = df.icustay_id
df_new_data = pd.concat([id_, df_new_data], axis=1)
df_new_data.rename(columns={340: 'IsReadmitted'}, inplace=True)
# df_new_data.head(2)
# df_new_data.to_csv("Data/new_train_data.csv", index=False)
# df_new_data.to_csv("Data/encode_test_data.csv", index=False)

clf = RandomForestClassifier(n_estimators=25,
                             max_depth=25,
                             min_samples_leaf=50)

train_data = df_new_data.iloc[:, 1:]
col = train_data.shape[1]
X, y = train_data.iloc[:, 0: col - 1], train_data.iloc[:, col - 1]
t = sum(cross_val_score(clf, X, y, cv=5)) / 5
print(t)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()